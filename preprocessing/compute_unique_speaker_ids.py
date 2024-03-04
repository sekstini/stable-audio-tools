import re
from pathlib import Path

import fire
import msgspec

from preprocessing.common import load_metadata_l


RE_INT = re.compile("\d+")
def ints(s: str) -> tuple[int]:
    return tuple(map(int, RE_INT.findall(s)))


def main(data_root_path: str | Path):
    data_root_path = Path(data_root_path)

    speaker_ids = set()
    total = 0

    for p in data_root_path.glob("**/*.json"):
        print(f"Loading speaker_ids from {p}")
        metadata = load_metadata_l(p)
        for sample in metadata:
            speaker_ids.add(sample.speaker_id)
            total += 1

    print(f"Unique speaker IDs: {len(speaker_ids)}")
    print(f"Total samples: {total}")
    print(f"Unique portion: {len(speaker_ids) / total:.2%}")

    speaker_ids = sorted(speaker_ids, key=lambda s: (s.split("/")[0], *ints(s)))
    mapping = {sid: i for i, sid in enumerate(speaker_ids)}

    Path("data/processed_other/speaker_id_mapping.json").write_bytes(msgspec.json.encode(mapping))
    Path("data/processed_other/speaker_ids.txt").write_text("\n".join(speaker_ids))


if __name__ == "__main__":
    fire.Fire(main)
