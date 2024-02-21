import zipfile
import os
import json
import gc
from pathlib import Path
from typing import TypedDict
from time import perf_counter_ns as pc

import fire
import tqdm
import numpy as np
from safetensors.numpy import save_file


KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB


def strip_suffixes(p: Path) -> Path:
    return p.parent / p.name.split(".")[0]


def pack(data: bytes, metadata: dict) -> np.ndarray:
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    header_len = np.uint32(len(metadata_bytes))

    return np.concatenate((
        np.asarray([header_len], dtype=np.uint32).view(np.uint8),
        np.frombuffer(metadata_bytes, dtype=np.uint8),
        np.frombuffer(data, dtype=np.uint8)
    ))


# Just here for reference
def unpack(sample: np.ndarray) -> tuple[np.ndarray, dict]:
    header_len = sample[0:4].view(np.uint32).item()
    metadata = json.loads(sample[4:4+header_len].tobytes())
    data = sample[4+header_len:]
    return data, metadata


def load_transcriptions(zf: zipfile.PyZipFile) -> dict[str, str]:
    transcriptions = {}
    for info in zf.filelist:
        if info.filename.endswith(".txt") and "eval" not in info.filename:
            lines = zf.read(info.filename).decode("utf-8").strip().split("\n")
            for line in lines:
                basename, text = line.split(" ", 1)
                transcriptions[basename] = text
    return transcriptions


def process_dataset(dataset_path: str | Path, max_byte_per_shard: int = 50*GiB):
    dataset_path = Path(dataset_path)
    out_dir = strip_suffixes(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_count = 1
    shard_idx = 0
    shard_state_dict = {}
    shard_byte_count = 0

    def save_shard():
        if shard_state_dict:
            save_file(shard_state_dict, out_dir / f"{shard_idx:08}.safetensors")
            gc.collect()

    with zipfile.PyZipFile(dataset_path) as zf:
        transcriptions = load_transcriptions(zf)

        last_split = None

        for info in tqdm.tqdm(zf.filelist):
            if not info.filename.endswith(".wav"):
                continue

            # hi-fi-captain/en-US/female/wav/train_non_parallel/Seikatsu03_E-SU_001230.wav
            _, language, gender, _, split, filename = info.filename.split("/")
            if split == "eval":
                print("Skipping eval: ", filename)
                continue

            basename = filename.removesuffix(".wav")
            text = transcriptions[basename]

            data = zf.read(info.filename)
            assert len(data) <= max_byte_per_shard

            if shard_byte_count + len(data) > max_byte_per_shard:
                save_shard()
                shard_idx += 1
                shard_state_dict = {}
                shard_byte_count = 0

            shard_byte_count += len(data)

            metadata = {"prompt": text, "speaker_id": f"hifi-captain/{language}/{gender}"}
            shard_state_dict[basename] = pack(data, metadata)

        save_shard()

    return shard_count


if __name__ == "__main__":
    fire.Fire(process_dataset)
