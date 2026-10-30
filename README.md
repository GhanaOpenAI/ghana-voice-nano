# Ghana Voice Nano

Small, fast, offline text-to-speech for Ghanaian languages, plus the full recipe to train
a new voice for another language.

- **Model:** [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) acoustic model (~18M parameters) +
  [Vocos](https://github.com/gemelo-ai/vocos) universal vocoder.
- **Runtime:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on CPU, ~13–16x realtime on 2 threads.
  The same files run on Linux/macOS/Windows, Android, iOS and the web, fully offline.
- **Text frontend:** espeak-ng. espeak has no Akan/Ewe/Ga voices; the Lingua Franca Nova voice
  (`lfn`) reads Latin-script Ghanaian orthography (incl. `ɛ ɔ ŋ`) as clean IPA.

| Voice | Model | Demo |
|---|---|---|
| Asante Twi | [ghanaopenai/ghana-voice-nano-twi](https://huggingface.co/ghanaopenai/ghana-voice-nano-twi) | [Space](https://huggingface.co/spaces/ghanaopenai/ghana-voice-nano-twi) |

```
ghana_voice_nano/   inference package (sherpa-onnx): Python API + CLI
recipe/             data prep -> training -> sherpa-onnx export (GPU)
serve/              HTTP API on Modal
space/              Hugging Face Space demo
mobile/             packaging + Android / iOS / Flutter / web guide
```

## Use a voice

```bash
pip install git+https://github.com/ghanaopenai/ghana-voice-nano
ghana-voice-nano --text "Mema wo akye. Wo ho te sɛn?" --out hello.wav
```

```python
from ghana_voice_nano import GhanaVoiceNano

tts = GhanaVoiceNano.from_pretrained("ghanaopenai/ghana-voice-nano-twi")   # downloads ~200 MB once
audio, sr = tts.synthesize("Awurade ne me hwɛfoɔ, biribiara renhia me.", speed=1.0)
tts.save("twi.wav", audio)

fast = GhanaVoiceNano.from_pretrained("ghanaopenai/ghana-voice-nano-twi", acoustic="steps2")
for chunk in tts.stream(long_text):   # one chunk per sentence, for low latency
    ...
```

`synthesize` returns float32 audio at 22.05 kHz. `speed > 1` speaks faster. Long text is split
into sentences and synthesized one at a time, so paragraphs stay stable.

**sherpa-onnx must be 1.x** (`>=1.12,<2`, pinned by the package). sherpa-onnx 2.0 dropped the
espeak-ng frontend for Matcha models.

## HTTP API (Modal)

```bash
pip install modal && modal setup
modal deploy serve/modal_app.py
curl "https://<your-workspace>--ghana-voice-nano-tts-web.modal.run/tts?text=Mema%20wo%20akye" -o hello.wav
curl -X POST ".../tts" -H "Content-Type: application/json" \
     -d '{"text": "Mema wo akye.", "speed": 1.0, "variant": "steps4"}' -o hello.wav
```

CPU-only containers, model baked into the image, scale to zero when idle. Set
`MODEL_REPO=<repo>` at deploy time to serve another voice.

## On-device

```bash
python mobile/package.py --model ghanaopenai/ghana-voice-nano-twi --variant steps4 --out dist/
```

See [mobile/README.md](mobile/README.md) for Android, iOS, Flutter and WebAssembly.

## Train a voice for another language

You need one speaker, ideally 5–20+ hours of clean speech with transcripts, as a Hugging Face
dataset (or local parquet files) with `audio` and `text` columns, and an NVIDIA GPU.

### 1. Setup

```bash
sudo apt-get install -y espeak-ng
git clone https://github.com/ghanaopenai/ghana-voice-nano && cd ghana-voice-nano
recipe/setup.sh              # .venv + the Matcha-TTS engine (pinned) in third_party/
source .venv/bin/activate
```

The default torch wheel is for CUDA 12.8; set `TORCH_INDEX_URL` for other drivers.

### 2. Pick the espeak voice

Check that espeak turns your language into sensible IPA:

```bash
espeak-ng -v lfn -q --ipa "Mema wo akye. Wo ho te sɛn?"
```

`lfn` is the tested choice for Latin-script Ghanaian languages. Use the same voice for prep
and export: the model is trained on espeak's output, and sherpa-onnx re-runs espeak on device.

### 3. Prepare data

```bash
python recipe/prep_data.py --dataset <hf-dataset> --subset <config> --out data/<lang> \
    --espeak-voice lfn --speaker-column speaker_id --speaker <id>   # if multi-speaker
```

This writes 22.05 kHz wavs, precomputed mels, `train.txt`/`val.txt` and `stats.json` (the mel
mean/std that training **and** export must use). Read `report.json`: `dropped_symbols` lists
characters espeak produced that the model cannot represent. A long list means the espeak voice
is a poor fit. Use `--dataset parquet --data-files 'path/*.parquet'` for local files, and
`--max-hours` to cap the size.

### 4. Train

```bash
hf download ghanaopenai/ghana-voice-nano-twi checkpoints/twi_epoch045.ckpt --local-dir pretrained
recipe/train.sh data/<lang> pretrained/checkpoints/twi_epoch045.ckpt
```

- Warm start from the Twi checkpoint for Ghanaian languages; it already knows the `lfn`
  phoneme set. Only weights are copied; mel statistics always come from your `stats.json`.
- Checkpoints go to `runs/train/nano/runs/<date>/checkpoints/` (best 3 by
  `sub_loss/val_diff_loss`, plus `last.ckpt`). Training stops after 5 epochs without
  improvement. Watch it with `tensorboard --logdir runs`.
- Extra Hydra overrides pass through, e.g. `data.batch_size=32 trainer.max_epochs=100`.
- Twi reference: ~128k clips, 45 epochs from English LJSpeech; val diff loss ~0.26.

### 5. Export

```bash
python recipe/export_sherpa.py --ckpt runs/train/nano/runs/<date>/checkpoints/<best>.ckpt \
    --out bundles/<lang> --name <lang> --language "<Language name>" --espeak-voice lfn --steps 4 2
ghana-voice-nano --model bundles/<lang> --text "..." --out test.wav
```

The bundle (acoustic `.onnx` per step count, Vocos vocoder, `tokens.txt`, slimmed
`espeak-ng-data/`, `config.json`) is ready for `hf upload`, the Space, the Modal API and the
mobile packager.

The exporter checks the checkpoint's `mel_mean`/`mel_std` buffers against the statistics the
model was trained with and fixes them if they differ. The first Twi release shipped with the
English LJSpeech values (copied in by an older warm-start loader), which compressed the mel
range and made every output hissy. Vocoder copy-synthesis was clean; the fix alone lowered
the noise floor by ~18 dB.

## Credits and licenses

Code: MIT. Built on [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) (MIT),
[Vocos](https://github.com/gemelo-ai/vocos) (MIT), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
(Apache-2.0) and [espeak-ng](https://github.com/espeak-ng/espeak-ng) (GPL-3.0; its data ships in
model bundles under that license).
