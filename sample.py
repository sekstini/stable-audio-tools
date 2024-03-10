import fire
import torch
import torchaudio
import librosa

from stable_audio_tools.training.utils import copy_state_dict
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.lm import AudioLanguageModelWrapper


def trim_silence(wav: torch.Tensor) -> torch.Tensor:
    wav = torch.from_numpy(librosa.effects.trim(wav.numpy())[0])
    return wav


def load_model(ckpt_path: str, device: str | torch.device = "cuda") -> tuple[AudioLanguageModelWrapper, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["state_dict"]
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    
    model = create_model_from_config(ckpt["model_config"])
    copy_state_dict(model, state_dict)
    model.to(device).eval().requires_grad_(False)

    return model, ckpt["model_config"]


def main(
    ckpt_path: str,
    
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    audio_prefix: str | None = None,

    batch_size: int = 1,
    max_gen_len: int = 512,
    
    speaker_id: int = 5,
    snr: float = 70.0,
    c50: float = 60.0,
    speaking_rate: float = 13.5,

    temperature: float = 1.0,
    top_p: float = 0.7,
    cfg_scale: float = 1.0,
    prepend_cond_cfg: float | None = None,
    global_cond_cfg: float | None = None,

    seed: int | None = None,
):
    assert audio_prefix is None, "audio_prefix is not supported yet"

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Prompt: {prompt}")
    print(f"{speaker_id=}, {snr=}, {c50=}, {speaking_rate=}")
    print(f"{temperature=}, {top_p=}, {cfg_scale=}")
    print(f"{batch_size=}, {max_gen_len=}")
    print(f"{seed=}")

    model, model_config = load_model(ckpt_path)
    sample_rate = model_config["sample_rate"]

    wavs = model.generate_audio(
        max_gen_len=max_gen_len,
        batch_size=batch_size,
        conditioning=[{
            "prompt": prompt,
            "speaker_id": speaker_id,
            "snr": snr,
            "c50": c50,
            "speaking_rate": speaking_rate,
        }] * batch_size,

        cfg_scale=cfg_scale,
        prepend_cond_cfg=prepend_cond_cfg,
        global_cond_cfg=global_cond_cfg,
        temp=temperature,
        top_p=top_p,
    )

    wavs = wavs.cpu()
    for i, wav in enumerate(wavs):
        wav = trim_silence(wav)
        torchaudio.save(f"output-{i:04}.wav", wav, sample_rate)

if __name__ == "__main__":
    fire.Fire(main)
