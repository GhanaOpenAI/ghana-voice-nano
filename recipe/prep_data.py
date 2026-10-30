#!/usr/bin/env python3
"""Prepare a Hugging Face speech dataset for Ghana Voice Nano (Matcha-TTS) training.

One speaker, one language. For each kept clip:
  wavs/<id>.wav    22.05 kHz mono PCM16
  mels/<id>.npy    raw (un-normalised) 80-band mel; the trainer loads these instead of
                   recomputing an FFT every step
and at the end:
  train.txt / val.txt   filelists "wav_path|ipa_phonemes"
  stats.json            mel_mean / mel_std over the training set (training MUST use these)
  report.json           counts, symbol coverage and samples of the phonemizer output

Phonemes come from espeak-ng with --espeak-voice. espeak has no Akan/Ewe/Ga/... voices; the
Lingua Franca Nova voice "lfn" reads Latin-script Ghanaian orthography (incl. ɛ, ɔ, ŋ) as
clean IPA and is what the Twi model uses. The same voice must be used at inference
(recipe/export_sherpa.py --espeak-voice), because sherpa-onnx re-runs espeak on device.

Example (Asante Twi from ghana-speech):
    python recipe/prep_data.py --dataset michsethowusu/ghana-speech --subset Asante_Twi_twi \
        --out data/twi --espeak-voice lfn --max-hours 60
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from phonemizer.backend import EspeakBackend
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from matcha.text.symbols import symbols
from matcha.utils.audio import mel_spectrogram

SR, N_FFT, N_MELS, HOP, WIN, F_MIN, F_MAX = 22050, 1024, 80, 256, 1024, 0, 8000


def get_args():
    p = argparse.ArgumentParser(description="HF speech dataset -> Matcha training data")
    p.add_argument("--dataset", required=True, help="HF dataset id")
    p.add_argument("--subset", default=None, help="dataset config name, e.g. Asante_Twi_twi")
    p.add_argument("--data-files", default=None,
                   help="local files instead of a Hub id, e.g. --dataset parquet --data-files 'shards/*.parquet'")
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default=None, help="use this split for validation instead of holding out")
    p.add_argument("--out", required=True)
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--text-column", default="text")
    p.add_argument("--speaker-column", default=None, help="keep only --speaker from this column")
    p.add_argument("--speaker", default=None)
    p.add_argument("--espeak-voice", default="lfn")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument("--max-hours", type=float, default=None, help="stop after this much training audio")
    p.add_argument("--n-val", type=int, default=200, help="held-out clips when --val-split is unset")
    p.add_argument("--drop-digits", action="store_true",
                   help="drop transcripts containing digits (their spoken form is ambiguous)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


class Phonemizer:
    def __init__(self, voice):
        self.backend = EspeakBackend(voice, preserve_punctuation=True, with_stress=True,
                                     language_switch="remove-flags")
        self.symbols = set(symbols)
        self.dropped = {}

    def __call__(self, texts):
        out = []
        for ph in self.backend.phonemize([t.lower() for t in texts], strip=True, njobs=1):
            kept = []
            for c in ph:
                if c in self.symbols:
                    kept.append(c)
                else:
                    self.dropped[c] = self.dropped.get(c, 0) + 1
            out.append(" ".join("".join(kept).split()))
        return out


def iter_split(args, split):
    kw = {"token": os.environ.get("HF_TOKEN")} if os.environ.get("HF_TOKEN") else {}
    if args.data_files:
        kw["data_files"] = args.data_files
    ds = load_dataset(args.dataset, args.subset, split=split, **kw)
    if args.speaker_column:
        ds = ds.filter(lambda s: str(s) == str(args.speaker), input_columns=[args.speaker_column])
    return ds.cast_column(args.audio_column, Audio(sampling_rate=SR))


def write_clips(args, ds, tag, out, phonemize, budget_s=None):
    """Resample, filter and phonemize one split; returns [(wav_path, phonemes, seconds)]."""
    rows, total, batch = [], 0.0, []

    def flush():
        for (path, secs, text), ph in zip(batch, phonemize([b[2] for b in batch])):
            if len(ph.replace(" ", "")) >= 2:
                rows.append((path, ph, secs))
        batch.clear()

    for i, ex in enumerate(tqdm(ds, desc=f"[prep {tag}]")):
        text = (ex[args.text_column] or "").strip()
        if not text or (args.drop_digits and any(c.isdigit() for c in text)):
            continue
        audio = np.asarray(ex[args.audio_column]["array"], dtype=np.float32)
        secs = len(audio) / SR
        if not args.min_seconds <= secs <= args.max_seconds:
            continue
        path = out / "wavs" / f"{tag}_{i:07d}.wav"
        if not path.exists():
            sf.write(path, audio, SR, subtype="PCM_16")
        batch.append((str(path.resolve()), secs, text))
        total += secs
        if len(batch) >= 256:
            flush()
        if budget_s and total >= budget_s:
            break
    flush()
    return rows


class MelJob(Dataset):
    def __init__(self, paths, mels_dir):
        self.paths, self.mels_dir = paths, mels_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        audio, _ = sf.read(self.paths[i], dtype="float32", always_2d=True)
        mel = mel_spectrogram(torch.from_numpy(audio.T), N_FFT, N_MELS, SR, HOP, WIN, F_MIN, F_MAX,
                              center=False).squeeze(0)
        np.save(self.mels_dir / f"{Path(self.paths[i]).stem}.npy", mel.numpy())
        # sum, sum of squares, frame count: enough for an exact global mean/std
        return torch.tensor([float(mel.sum()), float((mel.double() ** 2).sum()), float(mel.shape[-1])],
                            dtype=torch.float64)


def mels_and_stats(paths, train_paths, out, workers):
    mels_dir = out / "mels"
    loader = DataLoader(MelJob(paths, mels_dir), batch_size=32, num_workers=workers,
                        collate_fn=lambda b: torch.stack(b))
    train = {Path(p).stem for p in train_paths}
    acc = torch.zeros(3, dtype=torch.float64)
    seen = 0
    for batch in tqdm(loader, desc="[mels]"):
        for j, row in enumerate(batch):
            if Path(paths[seen + j]).stem in train:
                acc += row
        seen += len(batch)
    n = acc[2].item() * N_MELS
    mean = acc[0].item() / n
    return {"mel_mean": mean, "mel_std": math.sqrt(acc[1].item() / n - mean ** 2)}


def main():
    a = get_args()
    random.seed(a.seed)
    out = Path(a.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    (out / "mels").mkdir(exist_ok=True)
    phonemize = Phonemizer(a.espeak_voice)

    budget = a.max_hours * 3600 if a.max_hours else None
    rows = write_clips(a, iter_split(a, a.split), "train", out, phonemize, budget)
    if a.val_split:
        val = write_clips(a, iter_split(a, a.val_split), "val", out, phonemize)
        train = rows
    else:
        random.shuffle(rows)
        val, train = rows[: a.n_val], rows[a.n_val:]
    random.shuffle(train)

    for name, items in (("train.txt", train), ("val.txt", val)):
        (out / name).write_text("".join(f"{p}|{ph}\n" for p, ph, _ in items), encoding="utf-8")

    stats = mels_and_stats([p for p, _, _ in train + val], [p for p, _, _ in train], out, a.num_workers)
    stats["n_train"], stats["n_val"] = len(train), len(val)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    report = {
        "dataset": a.dataset, "subset": a.subset, "espeak_voice": a.espeak_voice,
        "train_hours": round(sum(s for _, _, s in train) / 3600, 2),
        "val_hours": round(sum(s for _, _, s in val) / 3600, 2),
        # Characters espeak produced that the model's symbol table cannot represent. A long
        # list here means the espeak voice is a poor fit for the language.
        "dropped_symbols": dict(sorted(phonemize.dropped.items(), key=lambda kv: -kv[1])),
        "examples": [ph for _, ph, _ in train[:5]],
        **stats,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
