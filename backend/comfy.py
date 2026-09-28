"""Isolated ComfyUI client and allowlisted Wan workflow builder."""
from __future__ import annotations

import copy
import json
import mimetypes
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import monotonic
from time import sleep
from urllib.parse import urlencode
from uuid import uuid4

import requests
try:
    import websocket
except ImportError:  # deployment can poll until requirements are refreshed
    websocket = None  # type: ignore[assignment]

BASE_URL = "http://127.0.0.1:8188"
WORKFLOWS = Path(__file__).with_name("workflows")
# Must match WanCameraEmbedding.camera_pose exactly; ComfyUI rejects the whole
# prompt for any value outside this list. Verified against the running install.
CAMERA_MOTIONS = (
    "Static",
    "Pan Up",
    "Pan Down",
    "Pan Left",
    "Pan Right",
    "Zoom In",
    "Zoom Out",
    "Anti Clockwise (ACW)",
    "ClockWise (CW)",
)
INTERPOLATION_MULTIPLIERS = frozenset({1, 2, 4})
OUTPUT_RESOLUTIONS = {"native": None, "720p": 720, "1080p": 1080}
VIDEO_PIXEL_BUDGET = 512 * 512
T2V_SIZES = {"landscape": (832, 480), "portrait": (480, 832), "square": (512, 512)}
LONG_READ_TIMEOUT_SECONDS = 1800

@dataclass(frozen=True, slots=True)
class VideoResult:
    prompt_id: str
    content: bytes
    filename: str
    duration_seconds: float
    poster_content: bytes
    poster_filename: str

def _node(graph: dict, *, class_type: str | None = None, title: str | None = None) -> dict:
    matches = [n for n in graph.values() if (class_type is None or n.get("class_type") == class_type) and (title is None or n.get("_meta", {}).get("title") == title)]
    if len(matches) != 1:
        raise ValueError(f"Workflow marker is not unique: {class_type or title}")
    return matches[0]

def derive_i2v_size(width: int, height: int) -> tuple[int, int]:
    """Fit a source aspect into the measured 512-square budget and Wan stride."""
    ratio = min(2.0, max(0.5, max(1, width) / max(1, height)))
    derived_height = max(16, int(math.sqrt(VIDEO_PIXEL_BUDGET / ratio)) // 16 * 16)
    derived_width = max(16, int(derived_height * ratio) // 16 * 16)
    while derived_width * derived_height > VIDEO_PIXEL_BUDGET:
        if derived_width >= derived_height: derived_width -= 16
        else: derived_height -= 16
    return derived_width, derived_height


def derive_upscale_size(width: int, height: int, output_resolution: str) -> tuple[int, int]:
    edge = OUTPUT_RESOLUTIONS[output_resolution]
    if edge is None: return width, height
    # The label is the vertical edge: conventional 720p/1080p for landscape,
    # and the long edge for portrait as promised by the UI.
    target_height = int(edge)
    target_width = max(16, round(target_height * width / height / 16) * 16)
    return target_width, target_height


def build_workflow(mode: str, *, prompt: str, negative_prompt: str, seed: int, source_filename: str | None = None, camera_motion: str = "Zoom In", interpolation: int = 2, output_resolution: str = "native", steps: int | None = None, generation_size: tuple[int, int] | None = None, orientation: str = "landscape") -> dict:
    if mode not in {"t2v", "i2v"}:
        raise ValueError("Unsupported video mode.")
    graph = json.loads((WORKFLOWS / f"wan21_{mode}.json").read_text(encoding="utf-8"))
    _node(graph, title="CLIP Text Encode (Positive Prompt)")["inputs"]["text"] = prompt
    _node(graph, title="CLIP Text Encode (Negative Prompt)")["inputs"]["text"] = negative_prompt
    _node(graph, class_type="KSampler")["inputs"]["seed"] = int(seed)
    if steps is not None:
        _node(graph, class_type="KSampler")["inputs"]["steps"] = int(steps)
    if interpolation not in INTERPOLATION_MULTIPLIERS:
        raise ValueError("Unsupported video interpolation mode.")
    if output_resolution not in OUTPUT_RESOLUTIONS:
        raise ValueError("Unsupported video output resolution.")
    if mode == "t2v":
        latent = _node(graph, class_type="EmptyHunyuanLatentVideo")["inputs"]
        native_size = T2V_SIZES.get(orientation)
        if native_size is None: raise ValueError("Unsupported video orientation.")
        latent.update(width=native_size[0], height=native_size[1], length=49)
    else:
        if not source_filename:
            raise ValueError("Image-to-video requires a source filename.")
        if camera_motion not in CAMERA_MOTIONS:
            raise ValueError("Unsupported camera motion.")
        _node(graph, class_type="LoadImage")["inputs"]["image"] = source_filename
        camera = _node(graph, class_type="WanCameraEmbedding")["inputs"]
        native_size = generation_size or (512, 512)
        camera.update(camera_pose=camera_motion, width=native_size[0], height=native_size[1], length=49)
    decoder_id = next(key for key, node in graph.items() if node.get("class_type") == "VAEDecode")
    create_id = next(key for key, node in graph.items() if node.get("class_type") == "CreateVideo")
    image_link: list[object] = [decoder_id, 0]
    next_id = max(map(int, graph)) + 1
    final_frame_id, save_frame_id = str(next_id), str(next_id + 1)
    graph[final_frame_id] = {
        "class_type": "ImageFromBatch",
        "inputs": {"image": image_link, "batch_index": 48, "length": 1},
        "_meta": {"title": "Select final native frame"},
    }
    graph[save_frame_id] = {
        "class_type": "SaveImage",
        "inputs": {"images": [final_frame_id, 0], "filename_prefix": "video/ForgeVID-last-frame"},
        "_meta": {"title": "Save final native frame"},
    }
    next_id += 2
    if interpolation > 1:
        loader_id, interpolation_id = str(next_id), str(next_id + 1)
        graph[loader_id] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": "rife_v4.26.safetensors"}, "_meta": {"title": "Load RIFE"}}
        graph[interpolation_id] = {"class_type": "FrameInterpolate", "inputs": {"interp_model": [loader_id, 0], "images": image_link, "multiplier": interpolation}, "_meta": {"title": "Interpolate frames"}}
        image_link = [interpolation_id, 0]
        next_id += 2
    target = derive_upscale_size(*native_size, output_resolution)
    if output_resolution != "native":
        loader_id, upscale_id, resize_id = str(next_id), str(next_id + 1), str(next_id + 2)
        graph[loader_id] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}, "_meta": {"title": "Load video upscaler"}}
        graph[upscale_id] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": [loader_id, 0], "image": image_link}, "_meta": {"title": "Upscale frames"}}
        graph[resize_id] = {"class_type": "ImageScale", "inputs": {"image": [upscale_id, 0], "upscale_method": "lanczos", "width": target[0], "height": target[1], "crop": "disabled"}, "_meta": {"title": "Resize video output"}}
        image_link = [resize_id, 0]
    graph[create_id]["inputs"]["images"] = image_link
    graph[create_id]["inputs"]["fps"] = 16.0 * interpolation
    return graph

class ComfyApiClient:
    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def upload_image(self, path: Path) -> str:
        with path.open("rb") as stream:
            response = requests.post(f"{self.base_url}/upload/image", files={"image": (path.name, stream, mimetypes.guess_type(path.name)[0] or "image/png")}, data={"overwrite": "true"}, timeout=90)
        response.raise_for_status()
        return str(response.json()["name"])

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/system_stats", timeout=3)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def generate_video(self, graph: dict, *, progress=None, on_submitted=None, cancel_event: Event | None = None, known_prompt_id: str | None = None) -> VideoResult:
        client_id = uuid4().hex
        started = monotonic()
        prompt_id = known_prompt_id
        if prompt_id is None:
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": copy.deepcopy(graph), "client_id": client_id}, timeout=30)
            response.raise_for_status(); body = response.json()
            if body.get("node_errors"):
                raise ValueError(f"ComfyUI rejected the workflow: {body['node_errors']}")
            prompt_id = str(body["prompt_id"])
            if on_submitted: on_submitted(prompt_id)
        ws = None
        if websocket:
            try:
                ws = websocket.create_connection(self.base_url.replace("http", "ws", 1) + f"/ws?clientId={client_id}", timeout=120)
            except (OSError, websocket.WebSocketException):
                ws = None
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    self.cancel(prompt_id); raise RuntimeError("Cancelled by user.")
                try:
                    message = json.loads(ws.recv()) if ws else {}
                except (OSError, ValueError, websocket.WebSocketException):
                    try: ws.close()
                    except Exception: pass
                    ws = None
                    message = {}
                data = message.get("data", {}) if isinstance(message, dict) else {}
                if data.get("prompt_id") not in (None, prompt_id): continue
                if message.get("type") == "progress" and progress:
                    progress(int(data.get("value", 0)), int(data.get("max", 1)), "sampling")
                if message.get("type") == "executing" and progress:
                    node = graph.get(str(data.get("node")), {})
                    kind = node.get("class_type")
                    stage = "interpolating" if kind == "FrameInterpolate" else "upscaling" if kind in {"ImageUpscaleWithModel", "ImageScale"} else "encoding" if kind in {"CreateVideo", "SaveVideo"} else None
                    if stage: progress(0, 1, stage)
                if message.get("type") == "execution_error": raise RuntimeError(str(data.get("exception_message") or "ComfyUI execution failed."))
                if message.get("type") == "execution_success": break
                history = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30).json()
                if history.get(prompt_id, {}).get("status", {}).get("completed"): break
                if not ws:
                    sleep(1)
        finally:
            if ws: ws.close()
        record = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30).json()[prompt_id]
        recorded_outputs = record.get("outputs", {})
        video_node_id = next(key for key, node in graph.items() if node.get("class_type") == "SaveVideo")
        poster_node_id = next(key for key, node in graph.items() if node.get("_meta", {}).get("title") == "Save final native frame")
        video_outputs = recorded_outputs.get(video_node_id, {}).get("images", [])
        poster_outputs = recorded_outputs.get(poster_node_id, {}).get("images", [])
        if not video_outputs: raise RuntimeError("ComfyUI completed without a video output.")
        if not poster_outputs: raise RuntimeError("ComfyUI completed without a final native frame.")
        meta = video_outputs[0]
        response = requests.get(f"{self.base_url}/view?{urlencode(meta)}", timeout=LONG_READ_TIMEOUT_SECONDS)
        response.raise_for_status()
        if "video" not in response.headers.get("content-type", ""): raise RuntimeError("ComfyUI returned a non-video artifact.")
        poster_meta = poster_outputs[0]
        poster_response = requests.get(f"{self.base_url}/view?{urlencode(poster_meta)}", timeout=LONG_READ_TIMEOUT_SECONDS)
        poster_response.raise_for_status()
        if "image" not in poster_response.headers.get("content-type", ""): raise RuntimeError("ComfyUI returned a non-image final frame.")
        return VideoResult(
            prompt_id,
            response.content,
            str(meta["filename"]),
            monotonic() - started,
            poster_response.content,
            str(poster_meta["filename"]),
        )

    def cancel(self, prompt_id: str) -> None:
        requests.post(f"{self.base_url}/interrupt", timeout=10).raise_for_status()
