from collections import defaultdict
import json
import atexit
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torchinfo
from torch import linalg as LA
from mamba_ssm import Mamba

from stable_audio_tools.models.mamba_lm import MambaModel

def now_str() -> str:
    """Returns a string of the current time, to be used in filenames"""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def attach_activation_monitoring_hooks(model: nn.Module, model_config: dict):
    hooks = []

    def should_hook_output(m: nn.Module, name: str) -> bool:
        if not any(name.startswith(s) for s in ("cond", "lm")):
            return False
        #if "prepend" in name:
        #    return True
        return isinstance(m, (nn.Embedding, nn.Linear, Mamba, MambaModel))

    run_folder = Path("activation_monitoring") / now_str()
    run_folder.mkdir(parents=True, exist_ok=True)

    summary_str = str(torchinfo.summary(model, depth=5, verbose=0))
    (run_folder / "model_summary.txt").write_text(summary_str)
    (run_folder / "model_config.json").write_text(json.dumps(model_config))

    outputs = defaultdict(list)
    inputs = defaultdict(list)

    def compute_stats(x: torch.Tensor) -> dict:
        x = x.cpu()
        #histogram, edges = x.histogram(bins=20, range=(x.min().item(), x.max().item()))
        return {
            #"histogram": histogram,
            #"histogram_edges": edges,
            #"min": x.min().item(),
            #"max": x.max().item(),
            #"mean": x.mean().item(),
            #"histogram": histogram,
            #"histogram_edges": (edges[0].item(), edges[-1].item()),
            "min": x.min().item(),
            "max": x.max().item(),
            "mean": x.mean().item(),
            "norm": LA.vector_norm(x, dim=-1).mean().item(),
            "shape": x.shape,
        }

    @torch.no_grad
    def record_layer_data(m: nn.Module, i, o, name: str):
        if isinstance(o, dict):
            for k, v in o.items():
                record_layer_data(m, i, v, name + "." + k)
        elif isinstance(o, (tuple, list)):
            for idx, v in enumerate(o):
                record_layer_data(m, i, v, name + f".{idx}")
        elif isinstance(o, torch.Tensor):
            outputs[name].append(compute_stats(o.detach()))
        else:
            print(f"Unknown output type {type(o)}")

    for name, m in model.named_modules():
        if should_hook_output(m, name):
            print(f"Attaching hook to {name}")
            hook_fn = partial(record_layer_data, name=name)
            handle = m.register_forward_hook(hook_fn)
            hooks.append(handle)

    def save_fn(inputs=inputs, outputs=outputs):
        torch.save(inputs, run_folder / "inputs.pt")
        torch.save(outputs, run_folder / "outputs.pt")

    atexit.register(save_fn)

    return hooks
