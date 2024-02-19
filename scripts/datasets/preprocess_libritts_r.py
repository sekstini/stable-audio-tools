import tarfile
import os
import json
from pathlib import Path

import fire
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


def process_dataset(dataset_path: str | Path, max_byte_per_shard: int = 25*GiB):
    dataset_path = Path(dataset_path)
    
    out_dir = dataset_path.with_suffix("").with_suffix("")
    out_dir.mkdir()
    
    texts = {}
    audio_basenames = set()
    with tarfile.open(dataset_path, mode="r|*") as stream:
        for info in stream:
            if info.name.endswith("normalized.txt"):
                text = stream.extractfile(info).read().decode("utf-8")
                basename = os.path.basename(info.name).removesuffix(".normalized.txt")
                texts[basename] = text
            elif info.name.endswith(".wav"):
                basename = os.path.basename(info.name).removesuffix(".wav")
                audio_basenames.add(basename)
            stream.members = []
    
    valid_basenames = texts.keys() & audio_basenames

    shard_idx = 0
    shard_state_dict = {}
    shard_byte_count = 0

    def save_shard():
        if shard_state_dict:
            save_file(shard_state_dict, out_dir / f"{shard_idx:08}.safetensors")
    
    with tarfile.open(dataset_path, mode="r|*") as stream:    
        for info in stream:
            if not info.name.endswith(".wav"):
                continue
                
            basename = os.path.basename(info.name).removesuffix(".wav")
            if basename not in valid_basenames:
                continue
            speaker_id = basename.split("_")[0]

            data = stream.extractfile(info).read()
            assert len(data) <= max_byte_per_shard
            
            if shard_byte_count + len(data) > max_byte_per_shard:
                save_shard()
                shard_idx += 1
                shard_state_dict = {}
                shard_byte_count = 0

            shard_byte_count += len(data)

            metadata = {"prompt": text, "speaker_id": speaker_id}
            shard_state_dict[basename] = pack(data, metadata)
                
            stream.members = []
        
        save_shard()
    
    return shard_idx + 1


if __name__ == "__main__":
    fire.Fire(process_dataset)
