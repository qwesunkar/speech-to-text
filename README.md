# speech-to-text

Three scripts, all running on an NVIDIA GPU:

- `stt.py` — transcribe audio/video files to text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- `tts.py` — turn the transcript back into speech in the original speaker's cloned voice with [XTTS v2](https://github.com/idiap/coqui-ai-TTS), each line at its original timestamp
- `separate.py` — split a song into vocals and beat with [Demucs](https://github.com/adefossez/demucs), so the lyrics can be re-voiced on the original beat

Plus `rvc/` — [Applio](https://github.com/IAHispano/Applio) (RVC) to train a singer's voice model and convert a vocal performance into that voice, keeping the flow — and `experiment/`, a test of how accurately it clones Young Thug.

## Setup

Requires Python 3.12, [uv](https://github.com/astral-sh/uv), `ffmpeg`, and an NVIDIA driver.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

The CUDA libraries come from pip, so no system CUDA install or `LD_LIBRARY_PATH` is needed. PyTorch is installed from the CUDA 12.8 index, which RTX 50xx GPUs require.

Models are downloaded on first run: whisper `large-v3-turbo` (~1.6 GB, `~/.cache/huggingface`) and XTTS v2 (~1.8 GB, `~/.local/share/tts`, or `~/snap/code/<rev>/.local/share/tts` when run from the VS Code snap terminal). The first `tts.py` run asks you to accept the XTTS license ([CPML](https://coqui.ai/cpml), non-commercial use only).

## Speech to text: `stt.py`

```bash
.venv/bin/python stt.py FILE [FILE ...] [--model NAME] [--language CODE] [--cpu]
```

| Option | Default | Description |
|---|---|---|
| `FILE` | — | One or more audio/video files (mp3, wav, m4a, mp4, mkv, ...) |
| `--model` | `large-v3-turbo` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `--language` | auto-detect | Language code, e.g. `en`, `ru` |
| `--cpu` | off | Run on CPU (int8) instead of GPU |

For each input it prints the transcript and writes, next to the input:

- `name.txt` — plain text, one segment per line
- `name.srt` — subtitles with timings

```bash
.venv/bin/python stt.py lecture.mp4
.venv/bin/python stt.py *.mp3 --language ru
.venv/bin/python stt.py "/path/with spaces/file.mp3"    # quote paths containing spaces
```

```
== speech.mp4  (language: en, 13s audio)
[   0.0] The quick brown fox jumps over the lazy dog.
[   3.9] Speech recognition turns spoken words into text.
-> speech.txt, speech.srt  (0.9s)
```

## Text to speech: `tts.py`

```bash
.venv/bin/python tts.py SRT VOICE --language CODE
```

| Argument | Description |
|---|---|
| `SRT` | `.srt` file made by `stt.py` (you can edit the text first) |
| `VOICE` | Original audio/video file to clone the voice from |
| `--beat` | Optional instrumental (e.g. `_beat.wav` from `separate.py`) to mix the speech onto |
| `--language` | Language of the text: `en`, `ru`, `de`, `fr`, `es`, `it`, `pt`, `pl`, `tr`, `nl`, `cs`, `ar`, `zh-cn`, `ja`, `ko`, `hu`, `hi` |

Writes `name_tts.mp3` next to the `.srt`. Each line starts at its original timestamp. If a line runs longer than the gap before the next one, it is sped up (at most 1.5×, shown as `(x1.13)` in the output).

Full round trip:

```bash
.venv/bin/python stt.py talk.mp3               # -> talk.txt, talk.srt
.venv/bin/python tts.py talk.srt talk.mp3 --language ru   # -> talk_tts.mp3
```

## Songs: `separate.py`

```bash
.venv/bin/python separate.py FILE [FILE ...]
```

Writes `name_vocals.wav` (voice only) and `name_beat.wav` (drums + bass + everything else) next to each input. The Demucs model (~80 MB) is downloaded on first run.

Re-voice a song on its own beat:

```bash
.venv/bin/python separate.py song.mp3              # -> song_vocals.wav, song_beat.wav
.venv/bin/python stt.py song_vocals.wav            # -> song_vocals.srt  (cleaner lyrics than from the full mix)
.venv/bin/python tts.py song_vocals.srt song_vocals.wav --language en --beat song_beat.wav
                                                   # -> song_vocals_tts.mp3
```

Using `song_vocals.wav` as the voice sample gives a much cleaner clone than the full song. The result is spoken, not rapped or sung: each line starts where the original did, but it doesn't follow the rhythm or melody within the line.

## Voice conversion: `rvc/` (RVC via Applio)

RVC changes *whose voice* a vocal performance is in, keeping its timing, flow and pitch. It can't create a performance: the words and rhythm always come from the input vocals. It needs a model trained on ~10+ min of the target voice.

Applio lives in `rvc/` with its **own venv** (it needs `transformers` 5, XTTS needs < 5). All commands below run from `rvc/`.

### Setup

```bash
git clone https://github.com/IAHispano/Applio.git rvc && cd rvc
uv venv .venv --python 3.12
VIRTUAL_ENV=$PWD/.venv uv pip install python-ffmpeg -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.venv/bin/python core.py prerequisites --pretraineds-hifigan --models --no-exe   # ~1.9 GB of base models
cp assets/config_template.json assets/config.json
git apply ../applio.patch   # made against Applio 55fe0b9
```

Two local changes are needed (the last two lines above):

- `assets/config.json` — Applio's web UI creates it on first launch; without it the CLI trains but silently never saves the final `.pth`.
- `rvc/train/train.py` — `DataLoader(num_workers=2, ..., prefetch_factor=2)` instead of `4` / `8`, so training fits in 14 GB of RAM. Close browsers etc. while training anyway.

The trainer hides its own errors; run it with `PYTHONUNBUFFERED=1` to see them.

### Train a voice

Put vocals-only files (from `separate.py`) of the target voice in one folder — solo songs only, since Demucs can't separate one rapper from another.

```bash
.venv/bin/python core.py preprocess --model-name NAME --dataset-path /path/to/vocals --sample-rate 40000 --cpu-cores 8
.venv/bin/python core.py extract --model-name NAME --sample-rate 40000 --cpu-cores 8 --gpu 0
PYTHONUNBUFFERED=1 .venv/bin/python core.py train --model-name NAME --sample-rate 40000 \
    --total-epoch 200 --save-every-epoch 25 --batch-size 8 --gpu 0
```

Output in `logs/NAME/`: `NAME_200e_<steps>s.pth` (the voice model) and `NAME.index`. On an RTX 5060, 20 min of vocals trains at ~19 s/epoch (200 epochs ≈ 65 min). Training also leaves `G_*.pth` / `D_*.pth` checkpoints (1.2 GB per save) — only needed to resume training; delete them to free space.

### Convert

```bash
.venv/bin/python core.py infer --input-path in_vocals.wav --output-path out.wav \
    --pth-path logs/NAME/NAME_200e_12000s.pth --index-path logs/NAME/NAME.index --index-rate 0.75 --pitch 0
```

`--pitch` shifts in semitones (e.g. `-4` to undo a +4 shift). A 3-min track converts in ~7 s on the GPU. Don't convert on the GPU while training — it runs out of GPU memory; `CUDA_VISIBLE_DEVICES=""` runs it on the CPU instead (~4.5 min, and slows training).

To put the result on a beat:

```bash
ffmpeg -i song_beat.wav -i out.wav -filter_complex \
  "[1]aresample=44100,pan=stereo|c0=c0|c1=c0[v];[0][v]amix=inputs=2:normalize=0,alimiter=limit=0.95" -b:a 192k out_on_beat.mp3
```

## Experiment: how accurately does RVC clone Young Thug?

Only `experiment/similarity.py` is in this repo. The songs, vocals, transcripts, outputs and the trained voice model are copyrighted or imitate a real artist, so they stay local (`experiment/songs/`, `dataset/`, `results/`, `rvc/logs/thug/`).

**Setup.** Trained on 6 songs (~20 min of vocals); `Twin Tower` held out as the test. Its real vocals were disguised (+4 semitones, formants shifted too, so it sounds like someone else), then converted back with RVC (`--pitch -4`) and compared to the real vocals.

**Voice score** — `similarity.py`: cosine similarity of [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) voiceprints, averaged over non-silent 3 s windows (1.0 = identical):

```bash
.venv/bin/python experiment/similarity.py REAL.wav FILE [FILE ...]
```

| vs. real Twin Tower vocals | Score |
|---|---|
| Thug on his 6 other songs (same person) | 0.55–0.70 |
| Disguised input | 0.30 |
| RVC, epochs 25–100 | 0.56–0.63 |
| RVC, epochs 125–200 | 0.62–0.66 |
| **RVC final (epoch 200)** | **0.65** |
| Final + index (rate 0.3 / 0.75) | 0.65 / 0.64 |
| XTTS reading of the lyrics → + RVC | 0.48 → 0.58 |

**Words** — share of the real-vocals transcript recovered when transcribing each version with `stt.py`:

| Version | Words matching |
|---|---|
| Disguised input | 50% |
| RVC epoch 25 | 10% |
| RVC final | 29% |
| XTTS reading (circular: it reads Whisper's own transcript) | 81% |

**Findings.** RVC recovers his voice to the same similarity as his own other songs, and most of that is reached by epoch 25. The index didn't change the score. But it loses articulation: on mumbled rap the converted words are much harder to recognise, improving with training (10% → 29%). XTTS reads words clearly but sounds less like him.

**Caveats.** One test song. The RVC version keeps his real performance underneath, which likely flatters its score. ECAPA was trained on speech, not rap vocals. Snapshot scores vary by about ±0.05, so small differences are noise.

## Notes

- `stt.py` skips silence and non-speech (voice activity detection), so a file with no speech gives empty `.txt`/`.srt` instead of hallucinated text. Music with singing can still produce junk lines such as `Субтитры сделал ...`.
- Existing output files are overwritten.
- With `--beat`, the mix is turned down if it would clip.
- Voice cloning needs at least ~6 s of clean speech in `VOICE`; only the first ~30 s is used. Short or noisy samples give garbled speech.
- XTTS occasionally adds a stray syllable at the end of a line; listen to the result.
- `tts.py` takes ~25 s for a short file, most of it loading the model.
