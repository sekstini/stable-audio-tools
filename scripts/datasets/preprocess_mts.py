import tarfile
import os
import json
import gc
from pathlib import Path
from typing import TypedDict
from time import perf_counter_ns as pc

import fire
import numpy as np
from safetensors.numpy import save_file


KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB

class PerSplitTranscriptions(TypedDict):
    train: dict[str, str]
    test: dict[str, str]
    dev: dict[str, str]


def strip_suffixes(p: Path) -> Path:
    return p.parent / p.name.split(".")[0]


def load_transcriptions(metadata_path: Path) -> PerSplitTranscriptions:
    out = {}
    for split in ["train", "test", "dev"]:
        transcripts_path = metadata_path / split / "transcripts.txt"
        d = {}
        for line in transcripts_path.open():
            key, text = line.split("\t", 1)
            d[key] = text
        print(split, len(d))
        out[split] = d
    return out


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


def process_dataset(dataset_path: str | Path, max_byte_per_shard: int = 50*GiB):
    dataset_path = Path(dataset_path)
    metadata_path = strip_suffixes(dataset_path)
    assert metadata_path.exists(), f"Read the instructions, metadata dir missing: {metadata_path}"

    # Just store the shards in the dir from the transcription extraction
    out_dir = metadata_path

    st = pc()
    per_split_transcriptions = load_transcriptions(metadata_path)
    et = pc()
    print(f"Load transcripts time: {et-st:>12,} ns")

    shard_count = 1
    shard_idx = 0
    shard_state_dict = {}
    shard_byte_count = 0

    def save_shard(split: str):
        if shard_state_dict:
            save_file(shard_state_dict, out_dir / split / f"{shard_idx:08}.safetensors")
            gc.collect()

    with tarfile.open(dataset_path, mode="r|*") as stream:
        last_split = None

        for i, info in enumerate(stream):
            if not info.name.endswith(".opus"):
                continue

            # mls_polish_opus/train/audio/6892/5541/6892_5541_000216.opus
            _, split, _, speaker_id, book_id, filename = info.name.split("/")

            if last_split is None:
                last_split = split
            elif last_split != split:
                save_shard(last_split)
                shard_idx = 0
                shard_count += 1
                shard_state_dict = {}
                shard_byte_count = 0
                last_split = split

            basename = filename[:filename.rfind('.')]
            text = per_split_transcriptions[split][basename]

            data = stream.extractfile(info).read()
            assert len(data) <= max_byte_per_shard

            if shard_byte_count + len(data) > max_byte_per_shard:
                save_shard(split)
                shard_idx += 1
                shard_state_dict = {}
                shard_byte_count = 0

            shard_byte_count += len(data)

            metadata = {"prompt": text, "speaker_id": speaker_id}
            shard_state_dict[basename] = pack(data, metadata)

            stream.members = []

        save_shard(split)

    return shard_count


if __name__ == "__main__":
    fire.Fire(process_dataset)
