"""
LibriTTS-R apparently has some missing text files. This script finds them and
deletes them. By default it just prints the paths of the missing files, but
you can pass --fix to delete them.
"""

from pathlib import Path

import fire


def main(
    libritts_r_path: str = "data/LibriTTS_R",
    fix: bool = False,
):
    suffixes = {".wav", ".txt"}
    paths = [p for p in Path(libritts_r_path).glob("**/*") if p.suffix in suffixes]

    txt_paths = {str(p) for p in paths if p.suffix == ".txt"}
    wav_paths = [p for p in paths if p.suffix == ".wav"]

    for wav_path in wav_paths:
        if str(wav_path.with_suffix(".normalized.txt")) in txt_paths:
            continue
        # elif str(wav_path.with_suffix(".original.txt")) in txt_paths:
        #    continue
        else:
            if fix:
                print(f"deleting: {wav_path}")
                wav_path.unlink()
            else:
                print(f"missing: {wav_path}")


if __name__ == "__main__":
    fire.Fire(main)
