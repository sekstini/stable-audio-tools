import tarfile
import os
from pathlib import Path

import fire
import numpy as np
from safetensors.numpy import save_file


KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB


def process_dataset(dataset_path: str | Path, max_byte_per_shard: int = 512*MiB):
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
            save_file(
                shard_state_dict,
                out_dir / f"{shard_idx:08}.safetensors",
                metadata={k: v for k, v in texts.items() if k in shard_state_dict},
            )
    
    with tarfile.open(dataset_path, mode="r|*") as stream:    
        for info in stream:
            if not info.name.endswith(".wav"):
                continue
                
            basename = os.path.basename(info.name).removesuffix(".wav")
            if basename not in valid_basenames:
                continue

            data = stream.extractfile(info).read()
            assert len(data) <= max_byte_per_shard
            
            if shard_byte_count + len(data) > max_byte_per_shard:
                save_shard()
                shard_idx += 1
                shard_state_dict = {}
                shard_byte_count = 0

            shard_byte_count += len(data)
            shard_state_dict[basename] = np.frombuffer(data, dtype=np.uint8)
                
            stream.members = []
        
        save_shard()
    
    return shard_idx + 1


if __name__ == "__main__":
    fire.Fire(process_dataset)
