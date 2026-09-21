import argparse
import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf
from TTS.api import TTS

SAMPLE_RATE = 24000  # XTTS output rate
MAX_SPEEDUP = 1.5    # how much a line may be sped up to fit before the next one starts


def srt_seconds(t):
    h, m, s = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_srt(path):
    # returns [(start, end, text), ...]
    segments = []
    for block in open(path, encoding="utf-8").read().strip().split("\n\n"):
        lines = block.strip().split("\n")
        start, end = lines[1].split(" --> ")
        segments.append((srt_seconds(start), srt_seconds(end), " ".join(lines[2:])))
    return segments


def main():
    parser = argparse.ArgumentParser(description="Re-voice an .srt with the original speaker's cloned voice (XTTS v2).")
    parser.add_argument("srt", help=".srt file made by stt.py")
    parser.add_argument("voice", help="original audio/video file to clone the voice from")
    parser.add_argument("--language", required=True, help="language code, e.g. en, ru")
    parser.add_argument("--beat", help="instrumental (e.g. _beat.wav from separate.py) to mix the speech onto")
    args = parser.parse_args()

    segments = read_srt(args.srt)

    # 1. load the model (downloaded once, then cached)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    with tempfile.TemporaryDirectory() as tmp:
        # 2. voice reference: decode any audio/video to mono wav
        ref = os.path.join(tmp, "ref.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", args.voice, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), ref], check=True)

        # 3. text -> speech, each line placed at its original start time
        out = np.zeros(int(segments[-1][1] * SAMPLE_RATE), dtype=np.float32)
        for i, (start, end, text) in enumerate(segments):
            wav = np.array(tts.tts(text=text, speaker_wav=ref, language=args.language), dtype=np.float32)

            # speed up (up to MAX_SPEEDUP) if the line runs into the next one
            slot = (segments[i + 1][0] if i + 1 < len(segments) else float("inf")) - start
            rate = len(wav) / SAMPLE_RATE / slot
            if rate > 1:
                wav = librosa.effects.time_stretch(wav, rate=min(rate, MAX_SPEEDUP))

            pos = int(start * SAMPLE_RATE)
            if pos + len(wav) > len(out):
                out = np.pad(out, (0, pos + len(wav) - len(out)))
            out[pos:pos + len(wav)] += wav
            print("[%6.1f] %s%s" % (start, text, "  (x%.2f)" % min(rate, MAX_SPEEDUP) if rate > 1 else ""))

    # 4. optionally mix onto the beat (resample speech to the beat's rate, same on every channel)
    sr = SAMPLE_RATE
    if args.beat:
        beat, sr = sf.read(args.beat, dtype="float32", always_2d=True)
        out = librosa.resample(out, orig_sr=SAMPLE_RATE, target_sr=sr)
        length = max(len(beat), len(out))
        out = np.pad(beat, ((0, length - len(beat)), (0, 0))) + np.pad(out, (0, length - len(out)))[:, None]
        out /= max(1.0, np.abs(out).max() / 0.95)  # turn the mix down instead of clipping

    # 5. save next to the .srt
    path = os.path.splitext(args.srt)[0] + "_tts.mp3"
    sf.write(path, np.clip(out, -1, 1), sr, format="MP3")
    print("->", path)


if __name__ == "__main__":
    main()
