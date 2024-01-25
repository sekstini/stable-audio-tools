import torch

from pathlib import Path


def get_custom_metadata(info: dict, _audio: torch.Tensor) -> dict:
    text_path = Path(info["path"]).with_suffix(".normalized.txt")
    return { "prompt": text_path.read_text("utf-8") }
