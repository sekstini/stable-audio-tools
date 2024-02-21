"""
Note: The original dataset has dev, train, and eval splits.
We don't use those, and just pack everything into a single shard.
"""

import tarfile
import os
import json
from pathlib import Path

import fire
import tqdm
import numpy as np
from safetensors.numpy import save_file


KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB


def pack(data: bytes, metadata: dict) -> np.ndarray:
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    header_len = np.uint32(len(metadata_bytes))

    return np.concatenate((
        np.asarray([header_len], dtype=np.uint32).view(np.uint8),
        np.frombuffer(metadata_bytes, dtype=np.uint8),
        np.frombuffer(data, dtype=np.uint8)
    ))


def load_transcriptions(dataset_path: Path) -> dict[str, str]:
    transcriptions = {}
    with tarfile.open(dataset_path, mode="r|*") as stream:
        for info in stream:
            if not info.name.endswith("read_transcriptions.txt"):
                stream.members = []
                continue

            lines = stream.extractfile(info).read().decode("utf-8").strip().split("\n")
            for line in lines:
                basename, text = line.split("\t", 1)
                transcriptions[basename] = text

            return transcriptions

    raise ValueError("No transcriptions found")


def process_dataset(dataset_path: str | Path, max_byte_per_shard: int = 50*GiB):
    dataset_path = Path(dataset_path)

    out_dir = dataset_path.with_suffix("")
    out_dir.mkdir(exist_ok=True)

    shard_idx = 0
    shard_state_dict = {}
    shard_byte_count = 0

    def save_shard():
        if shard_state_dict:
            save_file(shard_state_dict, out_dir / f"{shard_idx:08}.safetensors")

    print("Loading transcriptions...")
    transcriptions = load_transcriptions(dataset_path)

    with tarfile.open(dataset_path, mode="r|*") as stream:
        progress = tqdm.tqdm(total=len(transcriptions))
        for info in stream:
            if not info.name.endswith(".wav") or "conversational" in info.name:
                continue

            _, _, category, speaker_id, style, corpus, filename = info.name.split("/")

            basename = filename.removesuffix(".wav")
            text = transcriptions[basename]

            data = stream.extractfile(info).read()
            assert len(data) <= max_byte_per_shard

            if shard_byte_count + len(data) > max_byte_per_shard:
                save_shard()
                shard_idx += 1
                shard_state_dict = {}
                shard_byte_count = 0

            shard_byte_count += len(data)

            metadata = {"prompt": text, "speaker_id": speaker_id, "style": style}
            shard_state_dict[basename] = pack(data, metadata)

            progress.update()

            stream.members = []

        save_shard()

    return shard_idx + 1


if __name__ == "__main__":
    fire.Fire(process_dataset)
