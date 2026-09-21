import argparse
import os

import soundfile as sf
from demucs.api import Separator


def main():
    parser = argparse.ArgumentParser(description="Split songs into vocals and beat with Demucs.")
    parser.add_argument("files", nargs="+", help="audio or video files (mp3, wav, mp4, ...)")
    args = parser.parse_args()

    # 1. load the model (downloaded once, then cached)
    separator = Separator("htdemucs", progress=True)

    for path in args.files:
        # 2. song -> vocals, drums, bass, other
        _, stems = separator.separate_audio_file(path)
        beat = stems["drums"] + stems["bass"] + stems["other"]

        # 3. save next to the input
        base = os.path.splitext(path)[0]
        sf.write(base + "_vocals.wav", stems["vocals"].T.numpy(), separator.samplerate)
        sf.write(base + "_beat.wav", beat.T.numpy(), separator.samplerate)
        print(f"-> {base}_vocals.wav, {base}_beat.wav")


if __name__ == "__main__":
    main()
