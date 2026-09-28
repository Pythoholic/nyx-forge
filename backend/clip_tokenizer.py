"""Exact local CLIP BPE token counting used by the SDXL pipelines."""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as RustTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace


TOKENIZER_DATA_DIR = Path(__file__).with_name("tokenizer_data")


class CLIPBPETokenizer:
    """Swappable tokenizer implementation backed by the bundled CLIP files."""

    model_max_length = 77
    content_max_length = 75

    def __init__(self, data_dir: Path = TOKENIZER_DATA_DIR):
        self._tokenizer = RustTokenizer(BPE.from_file(
            str(data_dir / "vocab.json"),
            str(data_dir / "merges.txt"),
            end_of_word_suffix="</w>",
        ))
        self._tokenizer.pre_tokenizer = Whitespace()

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)


DEFAULT_TOKENIZER = CLIPBPETokenizer()
