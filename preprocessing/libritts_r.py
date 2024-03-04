import tarfile
import os
import io
from pathlib import Path
from typing import Iterator

import fire

from common import SampleStream, Sample


class LibriTTS_R_SampleStream(SampleStream):
    def __init__(self, dataset_path: Path):
        dataset_path = Path(dataset_path)

        subset = dataset_path.with_suffix("").stem.replace("_", "-")
        failed_examples_path = dataset_path.parent  / "libritts_r_failed_speech_restoration_examples" / f"{subset}_bad_sample_list.txt"
        failed_examples = set("LibriTTS_R" + name[1:] for name in failed_examples_path.read_text().strip().split("\n"))

        print(f"Found {len(failed_examples)} failed examples for {subset=}")

        prompts = {}
        audio_basenames = set()
        with tarfile.open(dataset_path, mode="r|*") as tar_stream:
            for info in tar_stream:
                if info.name in failed_examples:
                    continue
                if info.name.endswith("normalized.txt"):
                    text = tar_stream.extractfile(info).read().decode("utf-8")
                    basename = os.path.basename(info.name).removesuffix(".normalized.txt")
                    prompts[basename] = text
                elif info.name.endswith(".wav"):
                    basename = os.path.basename(info.name).removesuffix(".wav")
                    audio_basenames.add(basename)
                tar_stream.members = []

        self.valid_basenames = prompts.keys() & audio_basenames
        self.dataset_path = dataset_path
        self.prompts = prompts

    def __len__(self):
        return len(self.valid_basenames)

    def __iter__(self) -> Iterator[Sample]:
        with tarfile.open(self.dataset_path, mode="r|*") as tar_stream:
            for info in tar_stream:
                if not info.name.endswith(".wav"):
                    continue

                basename = os.path.basename(info.name).removesuffix(".wav")
                if basename not in self.valid_basenames:
                    continue

                audio_stream = io.BytesIO(tar_stream.extractfile(info).read())

                sample = Sample(
                    name=basename,
                    ext="wav",
                    prompt=self.prompts[basename],
                    speaker_id=f"LibriTTS-R/{basename.split('_')[0]}",
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

    sample_stream = LibriTTS_R_SampleStream(dataset_path)
    sample_stream.save_to_zip(out_path)


if __name__ == "__main__":
    fire.Fire(process_dataset)
