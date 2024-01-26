import json

import fire
import torch
import torchaudio
import torchinfo

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.lm import AudioLanguageModelWrapper
from stable_audio_tools.models.lm_backbone import MambaAudioLMBackbone
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict


def patch_mamba_with_cuda_graph(
    model: AudioLanguageModelWrapper,
    batch_size: int,
    max_seqlen: int,
    cfg_scale: float,
    decoding_seqlen=1,
    n_warmups: int = 2,
    dtype: torch.dtype = torch.float16,
    mempool=None,
):
    if cfg_scale != 1.0:
        batch_size *= 2

    model.lm.backbone.reset_generation_cache(max_seqlen, batch_size, dtype=dtype)
    inference_params = model.lm.backbone.inference_params
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32, device="cuda")

    hidden_dim = model.codebook_size
    x = torch.full((batch_size, decoding_seqlen, hidden_dim), 0.0, dtype=dtype, device="cuda")

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model.lm.backbone.forward(x, use_cache=True)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model.lm.backbone.forward(x, use_cache=True)

    # Patch the backbone forward to replay the graph
    def forward(self, new_x, **kwargs):
        x.copy_(new_x)
        graph.replay()
        return logits.clone()

    model.lm.backbone.forward = forward.__get__(model.lm.backbone)


def load_model(model_config_path: str, model_ckpt_path: str):
    model_config: dict = json.load(open(model_config_path))
    model: AudioLanguageModelWrapper = create_model_from_config(model_config) # type: ignore
    torchinfo.summary(model)

    print(f"Loading model checkpoint from {model_ckpt_path}")
    copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    model.pretransform.to("cuda").eval().requires_grad_(False)
    model.lm.to("cuda", torch.float16).eval().requires_grad_(False)

    return model, model_config

def main(
    model_config_path: str,
    model_ckpt_path: str = "exported_model.ckpt",

    seconds: float = 5.0,
    batch_size: int = 1,

    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    cfg_scale: float = 1.0,
):
    model, model_config = load_model(model_config_path, model_ckpt_path)

    frames_per_second = model.sample_rate / model.pretransform.downsampling_ratio # type: ignore

    max_seqlen = int(frames_per_second * seconds)

    if isinstance(model.lm.backbone, MambaAudioLMBackbone):
        patch_mamba_with_cuda_graph(model, batch_size, max_seqlen, cfg_scale)

    wavs = model.generate_audio(
        max_gen_len=max_seqlen,
        batch_size=batch_size,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        cfg_scale=cfg_scale,
    )

    print(f"Audio shape: {wavs.shape}")
    for i, wav in enumerate(wavs.cpu()):
        print(f"Saving audio to generated_{i}.wav")
        torchaudio.save(f"generated_{i}.wav", wav, model_config["sample_rate"])


if __name__ == "__main__":
    fire.Fire(main)
