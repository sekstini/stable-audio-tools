import zipfile
import io
from pathlib import Path
from typing import Iterator

import fire

from common import SampleStream, Sample


def load_transcriptions(zf: zipfile.PyZipFile) -> dict[str, str]:
    transcriptions = {}
    for info in zf.filelist:
        if info.filename.endswith(".txt") and "eval" not in info.filename:
            lines = zf.read(info.filename).decode("utf-8").strip().split("\n")
            for line in lines:
                basename, text = line.split(" ", 1)
                transcriptions[basename] = text
    return transcriptions


class HiFi_Captain_SampleStream(SampleStream):
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

        with zipfile.PyZipFile(dataset_path) as zf:
            self.transcriptions = load_transcriptions(zf)

    def __len__(self):
        return len(self.transcriptions)

    def __iter__(self) -> Iterator[Sample]:
        with zipfile.PyZipFile(self.dataset_path) as zf:
            for info in zf.filelist:
                if not info.filename.endswith(".wav"):
                    continue

                # hi-fi-captain/en-US/female/wav/train_non_parallel/Seikatsu03_E-SU_001230.wav
                _, language, gender, _, split, filename = info.filename.split("/")
                if split == "eval":
                    print("Skipping eval: ", filename)
                    continue

                basename = filename.removesuffix(".wav")
                audio_stream = io.BytesIO(zf.read(info.filename))

                sample = Sample(
                    name=basename,
                    ext="wav",
                    prompt=self.transcriptions[basename],
                    speaker_id=f"HiFi-Captain/{language}/{gender}",
                    audio_byte_stream=audio_stream,

                    gender=gender,
                    language=language.lower(),
                )

                yield sample


def process_dataset(dataset_path: str | Path):
    dataset_path = Path(dataset_path)
    processed_data_root = Path("data", "processed")
    processed_data_root.mkdir(exist_ok=True, parents=True)

    out_path = processed_data_root / dataset_path.name

    sample_stream = HiFi_Captain_SampleStream(dataset_path)
    sample_stream.save_to_zip(out_path)


if __name__ == "__main__":
    fire.Fire(process_dataset)
