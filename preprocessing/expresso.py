"""
Note: The original dataset has dev, train, and eval splits.
We don't use those, and just pack everything into a single shard.
"""

import tarfile
import io
from pathlib import Path
from typing import Iterator
from shutil import copyfileobj

import fire

from common import SampleStream, Sample


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


class Expresso_SampleStream(SampleStream):
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.transcriptions = load_transcriptions(dataset_path)

    def __len__(self):
        return len(self.transcriptions)

    def __iter__(self) -> Iterator[Sample]:
        with tarfile.open(self.dataset_path, mode="r|*") as stream:
            for info in stream:
                if not info.name.endswith(".wav") or "conversational" in info.name:
                    continue

                _, _, category, speaker_id, style, corpus, filename = info.name.split("/")

                basename = filename.removesuffix(".wav")

                audio_stream = io.BytesIO()
                copyfileobj(stream.extractfile(info), audio_stream)
                audio_stream.seek(0)

                sample = Sample(
                    name=basename,
                    ext="wav",
                    prompt=self.transcriptions[basename],
                    speaker_id=f"Expresso/{speaker_id}",
                    audio_byte_stream=audio_stream,

                    language="en-us",
                    custom={"style": style, "corpus": corpus, "category": category},
                )

                yield sample


def process_dataset(dataset_path: str | Path):
    dataset_path = Path(dataset_path)
    processed_data_root = Path("data", "processed")
    processed_data_root.mkdir(exist_ok=True, parents=True)

    out_path = processed_data_root / dataset_path.with_suffix(".zip").name

    sample_stream = Expresso_SampleStream(dataset_path)
    sample_stream.save_to_zip(out_path)


if __name__ == "__main__":
    fire.Fire(process_dataset)
