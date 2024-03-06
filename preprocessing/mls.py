import tarfile
import io
from pathlib import Path
from typing import Iterator, TypedDict
from shutil import copyfileobj

import fire

from common import SampleStream, Sample


class PerSplitTranscriptions(TypedDict):
    train: dict[str, str]
    test: dict[str, str]
    dev: dict[str, str]


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


def strip_suffixes(p: Path) -> Path:
    return p.parent / p.name.split(".")[0]


class Multilingual_LibriSpeech(SampleStream):
    def __init__(self, dataset_path: Path, split: str = "train"):
        metadata_path = strip_suffixes(dataset_path)
        assert metadata_path.exists(), f"Read the instructions, metadata dir missing: {metadata_path}"

        self.per_split_transcriptions = load_transcriptions(metadata_path)
        self.dataset_path = dataset_path
        self.split = split

    def __len__(self):
        return len(self.per_split_transcriptions[self.split])

    def __iter__(self) -> Iterator[Sample]:
        with tarfile.open(self.dataset_path, mode="r|*") as tar_stream:
            for info in tar_stream:
                if not info.name.endswith(".opus"):
                    continue

                # mls_polish_opus/train/audio/6892/5541/6892_5541_000216.opus
                _, split, _, speaker_id, book_id, filename = info.name.split("/")
                if split != self.split:
                    continue

                basename = filename[:filename.rfind('.')]
                audio_stream = io.BytesIO()
                copyfileobj(tar_stream.extractfile(info), audio_stream)
                audio_stream.seek(0)

                sample = Sample(
                    name=basename,
                    ext="opus",
                    prompt=self.per_split_transcriptions[split][basename],
                    speaker_id=f"MLS/{speaker_id}",
                    audio_byte_stream=audio_stream,
                )

                yield sample

                # Free memory
                tar_stream.members = []


def process_dataset(dataset_path: str | Path):
    dataset_path = Path(dataset_path)
    processed_data_root = Path("data", "processed")
    processed_data_root.mkdir(exist_ok=True, parents=True)

    out_path = processed_data_root / dataset_path.with_suffix("").with_suffix(".zip").name

    sample_stream = Multilingual_LibriSpeech(dataset_path, split="train")
    sample_stream.save_to_zip(out_path)


if __name__ == "__main__":
    fire.Fire(process_dataset)
