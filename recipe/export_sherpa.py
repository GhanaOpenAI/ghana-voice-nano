#!/usr/bin/env python3
"""Export a trained Matcha-TTS checkpoint to a sherpa-onnx TTS bundle.

Run inside the training environment (the Matcha-TTS fork must be importable).

Output bundle (everything sherpa-onnx needs, on desktop, server, Android, iOS and web):
    <out>/
      <name>_steps4.onnx, <name>_steps2.onnx   acoustic models (text ids -> mel), one per ODE step count
      vocos-22khz-univ.onnx                    universal vocoder (mel -> audio)
      tokens.txt                               symbol -> id table read by sherpa-onnx
      espeak-ng-data/                          phonemizer data, slimmed to the one espeak voice used
      config.json                              read by the ghana_voice_nano package

The checkpoint's mel_mean/mel_std buffers are checked against the statistics the model was
trained with (hyper_parameters.data_statistics) and corrected if they differ. The first
shipped Twi model had LJSpeech buffers copied in by a warm-start, which made every output noisy.

Usage:
    python recipe/export_sherpa.py --ckpt runs/.../checkpoints/last.ckpt --out bundle/ \
        --name twi --language "Asante Twi" --espeak-voice lfn --steps 4 2
"""
import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import onnx
import torch

from matcha.models.matcha_tts import MatchaTTS
from matcha.text.symbols import symbols

# Lightning checkpoints hold non-tensor objects; torch>=2.6 defaults to weights_only=True.
_torch_load = torch.load
torch.load = lambda *args, **kw: _torch_load(*args, **{**kw, "weights_only": False})

VOCOS_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx"
ESPEAK_DATA_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2"
SAMPLE_RATE = 22050


def get_args():
    p = argparse.ArgumentParser(description="Matcha checkpoint -> sherpa-onnx bundle")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True, help="file prefix, e.g. twi -> twi_steps4.onnx")
    p.add_argument("--language", required=True, help="human-readable language name")
    p.add_argument("--espeak-voice", default="lfn", help="espeak-ng voice used at data prep")
    p.add_argument("--steps", type=int, nargs="+", default=[4, 2], help="ODE step counts to export")
    p.add_argument("--author", default="GhanaOpenAI")
    p.add_argument("--dataset", default="")
    p.add_argument("--espeak-data", default=None,
                   help="espeak-ng-data dir (default: download the copy that matches sherpa-onnx)")
    return p.parse_args()


def load_model(ckpt):
    model = MatchaTTS.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    stats = model.hparams.data_statistics
    mean, std = float(stats["mel_mean"]), float(stats["mel_std"])
    if abs(float(model.mel_mean) - mean) > 1e-4 or abs(float(model.mel_std) - std) > 1e-4:
        print(
            f"[export] WARNING: checkpoint buffers mel_mean/std = {float(model.mel_mean):.6f}/"
            f"{float(model.mel_std):.6f} do not match training stats {mean:.6f}/{std:.6f}. "
            "Using the training stats."
        )
        model.mel_mean.fill_(mean)
        model.mel_std.fill_(std)
    print(f"[export] mel stats {mean:.6f}/{std:.6f}, n_spks={model.n_spks}, n_vocab={model.n_vocab}")
    return model, mean, std


def export_acoustic(model, path, n_timesteps):
    """ONNX with the sherpa-onnx Matcha signature: (x, x_lengths, scales) -> (mel, mel_lengths)."""

    def forward(x, x_lengths, scales):
        out = model.synthesise(x, x_lengths, n_timesteps, temperature=scales[0], length_scale=scales[1])
        return out["mel"], out["mel_lengths"]

    model.forward = forward
    x = torch.randint(1, 100, (1, 50), dtype=torch.long)
    inputs = (x, torch.tensor([50], dtype=torch.long), torch.tensor([0.667, 1.0], dtype=torch.float32))
    # The default dynamo exporter cannot trace the ODE solver loop; use the TorchScript one.
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=["x", "x_lengths", "scales"],
        output_names=["mel", "mel_lengths"],
        dynamic_axes={"x": {0: "batch_size", 1: "time"}, "x_lengths": {0: "batch_size"},
                      "mel": {0: "batch_size", 2: "time"}, "mel_lengths": {0: "batch_size"}},
        opset_version=15,
        do_constant_folding=True,
        dynamo=False,
    )


def add_metadata(path, meta):
    m = onnx.load(path)
    for k, v in meta.items():
        prop = m.metadata_props.add()
        prop.key, prop.value = k, str(v)
    onnx.save(m, path)


def write_tokens(path, n_vocab):
    """tokens.txt for sherpa-onnx: one "<symbol> <id>" per line, id == index in symbols.py.

    Two sherpa-onnx quirks:
      * symbols.py lists the apostrophe twice; sherpa needs unique symbols, so the first
        (unused) copy is written as a private-use placeholder character.
      * sherpa's espeak frontend wraps every utterance in ^...$, so those must be present
        even though the model never sees them (use_eos_bos=0).
    """
    vocab = symbols[:n_vocab]  # the symbol table may have extra entries (e.g. language tokens)
    last = {s: i for i, s in enumerate(vocab)}  # text_to_sequence maps a symbol to its last id
    lines, n_dup = [], 0
    for i, s in enumerate(vocab):
        if last[s] != i:
            s, n_dup = chr(0xE000 + n_dup), n_dup + 1
        lines.append(f"{s} {i}")
    lines += [f"^ {n_vocab}", f"$ {n_vocab + 1}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_espeak_data(explicit, cache_dir):
    """espeak-ng-data matching the espeak-ng engine compiled into sherpa-onnx.

    The system espeak-ng (used by phonemizer at data prep) is often an older release whose
    data files differ, so by default the official sherpa-onnx copy is downloaded.
    """
    if explicit:
        return explicit
    dst = Path(cache_dir) / "espeak-ng-data"
    if not dst.exists():
        tar = Path(cache_dir) / "espeak-ng-data.tar.bz2"
        tar.parent.mkdir(parents=True, exist_ok=True)
        print("[export] downloading sherpa-onnx espeak-ng-data")
        urllib.request.urlretrieve(ESPEAK_DATA_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(cache_dir)
        tar.unlink()
    return str(dst)


def copy_espeak_data(src, dst, voice):
    """Copy only what one voice needs (~0.7 MB instead of ~28 MB)."""
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in ["phondata", "phonindex", "phontab", "phondata-manifest", "intonations", f"{voice}_dict"]:
        shutil.copy(Path(src) / f, dst / f)
    voice_files = [p for p in Path(src, "lang").rglob(voice) if p.is_file()]
    if not voice_files:
        raise SystemExit(f"espeak voice '{voice}' not found under {src}/lang")
    rel = voice_files[0].relative_to(src)
    (dst / rel.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy(voice_files[0], dst / rel)


def main():
    a = get_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    epoch = torch.load(a.ckpt, map_location="cpu").get("epoch", "")

    acoustic = {}
    for steps in a.steps:
        # Fresh model per export: tracing fills the rotary-embedding cache for the dummy input
        # length, and a reused model would bake that cache into the next graph as a constant.
        model, mean, std = load_model(a.ckpt)
        path = out / f"{a.name}_steps{steps}.onnx"
        print(f"[export] {path.name} ({steps} ODE steps)")
        export_acoustic(model, str(path), steps)
        add_metadata(str(path), {
            "model_type": "matcha-tts", "language": a.language, "voice": a.espeak_voice,
            "has_espeak": 1, "jieba": 0, "n_speakers": 1, "sample_rate": SAMPLE_RATE,
            "version": 1, "pad_id": 0, "use_icefall": 0, "use_eos_bos": 0,
            "model_author": a.author, "dataset": a.dataset, "num_ode_steps": steps, "epoch": epoch,
            "mel_mean": f"{mean:.6f}", "mel_std": f"{std:.6f}",
        })
        acoustic[f"steps{steps}"] = path.name

    write_tokens(out / "tokens.txt", model.n_vocab)
    vocoder = out / "vocos-22khz-univ.onnx"
    if not vocoder.exists():
        print("[export] downloading vocos-22khz-univ.onnx")
        urllib.request.urlretrieve(VOCOS_URL, vocoder)
    copy_espeak_data(find_espeak_data(a.espeak_data, out.parent / ".espeak-cache"), out / "espeak-ng-data", a.espeak_voice)

    config = {
        "language": a.language, "espeak_voice": a.espeak_voice, "sample_rate": SAMPLE_RATE,
        "acoustic_models": acoustic, "default_acoustic": next(iter(acoustic)),
        "vocoder": vocoder.name, "tokens": "tokens.txt", "data_dir": "espeak-ng-data",
        "noise_scale": 0.667, "mel_mean": mean, "mel_std": std, "epoch": epoch,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[export] bundle ready: {out}")


if __name__ == "__main__":
    main()
