async function apiFetch(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    let detail = null;
    try {
      const body = await response.json();
      detail = body.detail ?? null;
      message = typeof body.detail === "string" ? body.detail : body.detail?.message || message;
    } catch {
      // Preserve the HTTP fallback when the response is not JSON.
    }
    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return response.json();
}

export const getModels = () => apiFetch("/api/models");
export const getBackends = () => apiFetch("/api/backends");
export const getBackendSettings = () => apiFetch("/api/settings/backends");
export const browseBackendPackage = () =>
  apiFetch("/api/settings/backends/browse", { method: "POST" });
export const updateBackend = (backendId, port, packageDir) =>
  apiFetch(`/api/backends/${encodeURIComponent(backendId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ port, package_dir: packageDir }),
  });
export const controlBackend = (backendId, operation) =>
  apiFetch(`/api/backends/${encodeURIComponent(backendId)}/${operation}`, { method: "POST" });
export const getBatchOptions = (params = {}) => {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value != null && value !== "") query.set(key, String(value));
  });
  return apiFetch(`/api/batch/options${query.size ? `?${query}` : ""}`);
};
export const previewBatch = (payload) =>
  apiFetch("/api/batch/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const createBatch = (payload) =>
  apiFetch("/api/batches", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const getBatch = (batchId) => apiFetch(`/api/batches/${batchId}`);
export const getBatches = (limit = 50) => apiFetch(`/api/batches?limit=${limit}`);
export const cancelBatch = (batchId) => apiFetch(`/api/batches/${batchId}/cancel`, { method: "POST" });
export const retryBatchFailures = (batchId) => apiFetch(`/api/batches/${batchId}/retry-failed`, { method: "POST" });
export const getAuthSession = () => apiFetch("/api/auth/session");
export const registerAccount = (displayName, email, password) =>
  apiFetch("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName, email, password }),
  });
export const loginAccount = (email, password) =>
  apiFetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
export const logoutAccount = () =>
  apiFetch("/api/auth/logout", { method: "POST" });
export const selectWorkspace = (workspace) =>
  apiFetch("/api/auth/workspace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ workspace }),
  });
export const updateAccountSettings = (maxActiveJobs) =>
  apiFetch("/api/auth/settings", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ max_active_jobs: maxActiveJobs }),
  });
export const getStorageSettings = () => apiFetch("/api/settings/storage");
export const setStorageSettings = (dataDir) =>
  apiFetch("/api/settings/storage", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data_dir: dataDir }),
  });
export const beginAuthenticatorSetup = (password) =>
  apiFetch("/api/auth/authenticator/setup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
export const confirmAuthenticatorSetup = (code) =>
  apiFetch("/api/auth/authenticator/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
export const loginWithAuthenticator = (code) =>
  apiFetch("/api/auth/authenticator/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
export const getPromptModels = () => apiFetch("/api/prompt-models");
export const testCloudCredential = (provider, apiKey) =>
  apiFetch("/api/cloud/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, api_key: apiKey }),
  });
export const getHistory = (beforeId = null, limit = 60) => {
  const query = new URLSearchParams({ limit: String(limit) });
  if (beforeId != null) query.set("before_id", String(beforeId));
  return apiFetch(`/api/history?${query.toString()}`);
};
export const getVideoHistory = (beforeId = null, limit = 60) => {
  const query = new URLSearchParams({ limit: String(limit) });
  if (beforeId != null) query.set("before_id", String(beforeId));
  return apiFetch(`/api/video/history?${query}`);
};
export const getVideoStatus = () => apiFetch("/api/video/status");
export const generateVideo = (mode, payload) =>
  apiFetch(`/api/video/${mode}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
export const generateVideoPrompt = (payload) =>
  apiFetch("/api/video/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
export const setVideoFavorite = (videoId, favorite) =>
  apiFetch(`/api/video/${videoId}/favorite`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ favorite }) });
export const deleteVideo = (videoId) => fetch(`/api/video/${videoId}`, { method: "DELETE", credentials: "same-origin" }).then((response) => { if (!response.ok) throw new Error(`Request failed (HTTP ${response.status}).`); });
export const getAnalytics = (days = 30) => apiFetch(`/api/analytics?days=${days}`);
export const getGenerationProgress = (requestId) =>
  apiFetch(`/api/generate/progress?request_id=${encodeURIComponent(requestId)}`);
export const getJob = (jobId, accessToken = null) => {
  const query = accessToken ? `?access=${encodeURIComponent(accessToken)}` : "";
  return apiFetch(`/api/jobs/${jobId}${query}`);
};
export const getJobs = (beforeId = null, limit = 50, status = null, kinds = null, mine = false) => {
  const query = new URLSearchParams({ limit: String(limit) });
  if (beforeId != null) query.set("before_id", String(beforeId));
  if (status) query.set("status", status);
  for (const kind of kinds ?? []) query.append("kind", kind);
  if (mine) query.set("mine", "true");
  return apiFetch(`/api/jobs?${query.toString()}`);
};
export const rerunJob = (jobId, accessToken = null) => {
  const query = accessToken ? `?access=${encodeURIComponent(accessToken)}` : "";
  return apiFetch(`/api/jobs/${jobId}/rerun${query}`, { method: "POST" });
};
export const setFavorite = (generationId, favorite) =>
  apiFetch(`/api/generations/${generationId}/favorite`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ favorite }),
  });
export const deleteJobs = (jobIds) =>
  apiFetch("/api/jobs/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_ids: jobIds }),
  });
export const cancelJob = (jobId, accessToken = null) => {
  const query = accessToken ? `?access=${encodeURIComponent(accessToken)}` : "";
  return apiFetch(`/api/jobs/${jobId}/cancel${query}`, { method: "POST" });
};
export const getGenerationLineage = (generationId, accessToken = null) => {
  const query = accessToken ? `?access=${encodeURIComponent(accessToken)}` : "";
  return apiFetch(`/api/generations/${generationId}/lineage${query}`);
};
export const rateImage = (generationId, score, reasons = [], accessToken = null) =>
  apiFetch(`/api/generations/${generationId}/rating`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ score, reasons, access_token: accessToken }),
  });
export const upscaleImage = (generationId, superRes = false, accessToken = null, requestId = null) =>
  apiFetch(`/api/generations/${generationId}/upscale`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      super_res: superRes,
      access_token: accessToken,
      request_id: requestId,
    }),
  });
export const pixelUpscaleImage = (generationId, multiplier, accessToken = null) =>
  apiFetch(`/api/generations/${generationId}/pixel-upscale`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ multiplier, access_token: accessToken }),
  });
export const getSurprisePrompt = (model, style, contentRating, aspectRatio, qualityMode, promptEngine, creativityLevel, orientation, ollamaModel, cloudProvider, cloudModel, cloudApiKey) =>
  apiFetch("/api/surprise", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      style,
      content_rating: contentRating,
      aspect_ratio: aspectRatio,
      quality_mode: qualityMode,
      high_res: qualityMode !== "normal",
      super_res: ["super", "8k", "12k"].includes(qualityMode),
      prompt_engine: promptEngine,
      creativity_level: creativityLevel,
      orientation,
      ollama_model: ollamaModel || null,
      cloud_provider: cloudProvider || null,
      cloud_model: cloudModel || null,
      cloud_api_key: cloudApiKey || null,
    }),
  });
export const generateImage = (payload) =>
  apiFetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      quality_mode: payload.resolution_mode,
      high_res: payload.resolution_mode !== "normal",
      super_res: ["super", "8k", "12k"].includes(payload.resolution_mode),
    }),
  });
export const transformImage = (payload) =>
  apiFetch("/api/img2img", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const uploadImage = (file, onProgress = null) => {
  const body = new FormData();
  body.append("file", file);
  if (!onProgress) return apiFetch("/api/uploads", { method: "POST", body });
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/uploads");
    request.withCredentials = true;
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100));
    };
    request.onload = () => {
      let response = {};
      try { response = JSON.parse(request.responseText || "{}"); } catch { /* use HTTP fallback */ }
      if (request.status >= 200 && request.status < 300) resolve(response);
      else reject(new Error(response.detail || `Request failed (HTTP ${request.status}).`));
    };
    request.onerror = () => reject(new Error("The upload could not be sent."));
    request.send(body);
  });
};
export const uploadImageFromUrl = (url) =>
  apiFetch("/api/uploads/from-url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
export const getIdentityProfiles = () => apiFetch("/api/identity-profiles");
export const suggestIdentityCrop = (payload) =>
  apiFetch("/api/identity-profiles/suggest-crop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const validateIdentityCrop = (payload) =>
  apiFetch("/api/identity-profiles/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const createIdentityProfile = (payload) =>
  apiFetch("/api/identity-profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const renameIdentityProfile = (profileId, name) =>
  apiFetch(`/api/identity-profiles/${profileId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
export const deleteIdentityProfile = (profileId) =>
  apiFetch(`/api/identity-profiles/${profileId}`, { method: "DELETE" });
export const proposeImageMasks = (payload) =>
  apiFetch("/api/img2img/mask-proposals", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const suggestImageTransform = (payload) =>
  apiFetch("/api/img2img/suggestion", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const surpriseGenerateImage = (model, style, contentRating, aspectRatio, qualityMode, promptEngine, creativityLevel, orientation, ollamaModel, cloudProvider, cloudModel, cloudApiKey, requestId, cfgScale) =>
  apiFetch("/api/surprise-generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      style,
      content_rating: contentRating,
      aspect_ratio: aspectRatio,
      quality_mode: qualityMode,
      high_res: qualityMode !== "normal",
      super_res: ["super", "8k", "12k"].includes(qualityMode),
      prompt_engine: promptEngine,
      creativity_level: creativityLevel,
      orientation,
      ollama_model: ollamaModel || null,
      cloud_provider: cloudProvider || null,
      cloud_model: cloudModel || null,
      cloud_api_key: cloudApiKey || null,
      request_id: requestId,
      cfg_scale: cfgScale ?? null,
    }),
  });
