import zipfile
import io
import mmap
from typing import Iterator, Literal, Optional
from abc import abstractmethod
from pathlib import Path
from multiprocessing.pool import Pool

import soundfile
import msgspec
from tqdm import tqdm
from g2p_en import G2p


class SampleMetadatav1(msgspec.Struct, gc=False):
    prompt: str
    speaker_id: str
    duration: float
    phonemes: list[str]

    gender: Optional[Literal["male", "female"]] = None

    # ISO 639-1 language code, lowercase. example: "en-us"
    language: Optional[str] = None

    custom: Optional[dict[str, str]] = None


class SampleMetadata(msgspec.Struct, gc=False):
    prompt: str
    speaker_id: str
    duration: float
    speaking_rate: Optional[float] = None
    snr: Optional[float] = None
    c50: Optional[float] = None


class SampleMetadataL(msgspec.Struct, gc=False, array_like=True):
    key: str
    prompt: str
    speaker_id: str
    duration: float
    speaking_rate: float
    snr: float
    c50: float


Metadatav1 = dict[str, SampleMetadatav1]
Metadata = dict[str, SampleMetadata]
MetadataL = list[SampleMetadataL]


_decode_metadatav1 = msgspec.json.Decoder(Metadatav1).decode
def load_metadatav1(p: Path) -> Metadatav1:
    with open(p, "rb") as f:
        return _decode_metadatav1(f.read())

_decode_metadata = msgspec.json.Decoder(Metadata).decode
def load_metadata(p: Path) -> Metadata:
    with open(p, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return _decode_metadata(mm)

_decode_metadata_l = msgspec.json.Decoder(MetadataL).decode
def load_metadata_l(p: Path) -> MetadataL:
    with open(p, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return _decode_metadata_l(mm)


class Sample(msgspec.Struct, gc=False):
    name: str
    ext: str
    prompt: str
    speaker_id: str
    audio_byte_stream: io.BytesIO

    gender: Optional[Literal["male", "female"]] = None
    language: Optional[str] = None

    custom: Optional[dict[str, str]] = None

global g2p
g2p = None

def _init_g2p():
    global g2p
    g2p = G2p()

def _compute_phonemes_for_sample(job: tuple[str, str]) -> tuple[str, list[str]]:
    global g2p
    basename, prompt = job
    phonemes = g2p(prompt)
    return basename, phonemes

class SampleStream:
    @abstractmethod
    def __init__(self, dataset_path: Path):
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        ...

    def __len__(self) -> int:
        ...

    def _validate_sample(self, sample: Sample):
        assert sample.gender in (None, "male", "female")
        assert sample.language is None or sample.language.islower()

    def _compute_phonemes(self, metadata: dict[str, SampleMetadatav1]):
        with Pool(initializer=_init_g2p) as pool:
            jobs = [(basename, m.prompt) for basename, m in metadata.items()]
            results = pool.map(_compute_phonemes_for_sample, tqdm(jobs, desc="Computing phonemes"))

        for basename, phonemes in results:
            metadata[basename].phonemes = phonemes


    def save_to_zip(self, zip_path: Path, write_metadata: bool = True):
        print("WARNING: OVERRIDING write_metadata to False temporarily.")
        write_metadata = False

        with zipfile.ZipFile(zip_path, mode="x") as zf:
            metadata: Metadata = {}
            num_samples = len(self) if hasattr(self, "__len__") else None

            for sample in tqdm(self, desc="Writing to zip", total=num_samples):
                audio_info = soundfile.info(sample.audio_byte_stream)
                sample.audio_byte_stream.seek(0)

                with zf.open(f"{sample.name}.{sample.ext}", mode="w") as f:
                    f.write(sample.audio_byte_stream.getbuffer())

                if write_metadata:
                    self._validate_sample(sample)

                    metadata[sample.name] = SampleMetadatav1(
                        prompt=sample.prompt,
                        phonemes=[],
                        speaker_id=sample.speaker_id,
                        duration=audio_info.duration,
                        gender=sample.gender,
                        language=sample.language,
                        custom=sample.custom,
                    )

        if write_metadata:
            # Save metadata without phonemes in case of failure
            partial_metadata_path = zip_path.with_suffix(".partial.json")
            partial_metadata_path.write_bytes(msgspec.json.encode(metadata))

            self._compute_phonemes(metadata)

            zip_path.with_suffix(".json").write_bytes(msgspec.json.encode(metadata))

            # Since we got this far, we can safely remove the partial metadata
            partial_metadata_path.unlink(missing_ok=True)
