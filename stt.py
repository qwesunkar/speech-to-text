import argparse
import ctypes
import glob
import os
import time

from faster_whisper import WhisperModel


def load_cuda_libs():
    # ctranslate2 needs cuBLAS + cuDNN; load the pip-installed ones so no LD_LIBRARY_PATH is needed
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
    for pkg in (nvidia.cublas.lib, nvidia.cudnn.lib):
        for lib in sorted(glob.glob(os.path.join(pkg.__path__[0], "*.so*"))):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


def srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files with faster-whisper.")
    parser.add_argument("files", nargs="+", help="audio or video files (mp3, wav, mp4, ...)")
    parser.add_argument("--model", default="large-v3-turbo", help="whisper model (default: large-v3-turbo)")
    parser.add_argument("--language", default=None, help="language code, e.g. en, ru (default: auto-detect)")
    parser.add_argument("--cpu", action="store_true", help="run on CPU (int8) instead of GPU")
    args = parser.parse_args()

    # 1. load the model (downloaded once, then cached)
    if args.cpu:
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
    else:
        load_cuda_libs()
        model = WhisperModel(args.model, device="cuda", compute_type="float16")

    for path in args.files:
        # 2. speech -> text
        start = time.time()
        segments, info = model.transcribe(path, language=args.language, vad_filter=True)
        print(f"\n== {path}  (language: {info.language}, {info.duration:.0f}s audio)")

        # 3. print it and save .txt + .srt next to the input
        base = os.path.splitext(path)[0]
        with open(base + ".txt", "w") as txt, open(base + ".srt", "w") as srt:
            for i, seg in enumerate(segments, 1):
                text = seg.text.strip()
                print("[%6.1f] %s" % (seg.start, text))
                txt.write(text + "\n")
                srt.write(f"{i}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{text}\n\n")
        print(f"-> {base}.txt, {base}.srt  ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
