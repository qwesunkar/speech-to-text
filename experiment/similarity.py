import argparse

import librosa
import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

SR = 16000   # ECAPA input rate
WINDOW = 3   # seconds per voiceprint window


def voiceprint(model, path):
    # average voiceprint over all non-silent 3 s windows
    y, _ = librosa.load(path, sr=SR, mono=True)
    step = WINDOW * SR
    windows = [y[i:i + step] for i in range(0, len(y) - step, step)]
    loudest = max(np.sqrt(np.mean(w ** 2)) for w in windows)
    voiced = [w for w in windows if np.sqrt(np.mean(w ** 2)) > 0.1 * loudest]
    with torch.no_grad():
        emb = model.encode_batch(torch.tensor(np.stack(voiced))).squeeze(1)
    emb = torch.nn.functional.normalize(emb, dim=1).mean(0)
    return torch.nn.functional.normalize(emb, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Voice similarity of each file to a reference (cosine of ECAPA voiceprints).")
    parser.add_argument("reference", help="the real voice")
    parser.add_argument("files", nargs="+", help="files to compare against the reference")
    args = parser.parse_args()

    model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    ref = voiceprint(model, args.reference)
    for path in args.files:
        print("%.3f  %s" % (float(ref @ voiceprint(model, path)), path))


if __name__ == "__main__":
    main()
