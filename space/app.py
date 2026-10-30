"""Hugging Face Space demo for Ghana Voice Nano (sherpa-onnx on CPU)."""
import json
import os
import time

import gradio as gr
import numpy as np

from ghana_voice_nano import GhanaVoiceNano, download

MODEL_REPO = os.environ.get("MODEL_REPO", "ghanaopenai/ghana-voice-nano-twi")
NUM_THREADS = 2
MAX_CHARS = 5000


def _read(path):
    try:
        return open(path).read().strip()
    except Exception:
        return None


def _hardware():
    """vCPU and RAM the container is allocated (cgroup limits), not the host's totals."""
    cpu_model = "CPU"
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    quota = None
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2 and not v2.startswith("max"):
        q, p = v2.split()[:2]
        quota = float(q) / float(p)
    mem = None
    v = _read("/sys/fs/cgroup/memory.max")
    if v and v != "max" and int(v) < (1 << 62):
        mem = int(v) / 2**30
    cpu = f"{quota:.0f} vCPU" if quota else f"{NUM_THREADS} threads"
    ram = f" · {mem:.0f} GB RAM" if mem else ""
    return f"{cpu} · {cpu_model}{ram} ({NUM_THREADS} synthesis threads)"


HW = _hardware()
MODEL_DIR = download(MODEL_REPO)
_base = GhanaVoiceNano(MODEL_DIR, num_threads=NUM_THREADS)
LABELS = {"steps4": "Balanced (4 steps), best quality", "steps2": "Fast (2 steps), lower quality"}
ENGINES = {LABELS.get(v, v): (_base if v == _base.config["default_acoustic"]
                              else GhanaVoiceNano(MODEL_DIR, acoustic=v, num_threads=NUM_THREADS))
           for v in _base.variants}
LANGUAGE = _base.config["language"]

SHORT = [
    "Awurade ne me hwɛfoɔ, biribiara renhia me.",
    "Mema wo akye. Wo ho te sɛn?",
    "Ghana yɛ ɔman a ɛwɔ Afrika atɔeɛ fam.",
]
with open(os.path.join(os.path.dirname(__file__), "long_samples.json"), encoding="utf-8") as f:
    LONG = json.load(f)


def synth(text, quality, speed):
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        return None, ""
    t0 = time.time()
    audio, sr = ENGINES[quality].synthesize(text, speed=float(speed))
    gen = time.time() - t0
    secs = len(audio) / sr
    stats = (
        f"**{secs:.1f}s** of audio in **{gen:.2f}s**, "
        f"**{secs / max(gen, 1e-9):.1f}x faster than realtime** · {len(text)} chars\n\n_Hardware: {HW}_"
    )
    return (sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16)), stats


with gr.Blocks(title=f"Ghana Voice Nano · {LANGUAGE}") as demo:
    gr.Markdown(
        f"# Ghana Voice Nano · {LANGUAGE} text-to-speech\n"
        "Small, fast, offline TTS: a Matcha-TTS acoustic model and Vocos vocoder running on "
        "**sherpa-onnx** (CPU). The same files run on Android, iOS and in the browser. Being "
        "non-autoregressive, it stays stable across whole paragraphs; try the long samples below.\n\n"
        f"Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) · "
        "Code, training recipe and mobile guide: "
        "[github.com/ghanaopenai/ghana-voice-nano](https://github.com/ghanaopenai/ghana-voice-nano)\n\n"
        f"Running on: **{HW}**"
    )
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label=f"{LANGUAGE} text", value=SHORT[0], lines=6, max_length=MAX_CHARS)
            quality = gr.Radio(list(ENGINES), value=list(ENGINES)[0], label="Quality / speed")
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speaking rate (higher = faster)")
            btn = gr.Button("Synthesize", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Speech")
            stats = gr.Markdown()
    btn.click(synth, [txt, quality, speed], [out, stats])

    gr.Markdown("#### Short phrases")
    gr.Examples(examples=[[s] for s in SHORT], inputs=[txt])
    gr.Markdown("#### Long text (paragraphs)")
    gr.Examples(
        examples=[[s["twi"]] for s in LONG],
        inputs=[txt],
        example_labels=[f"{s['num_sentences']} sentences · {s['chars']} chars" for s in LONG],
    )

if __name__ == "__main__":
    demo.launch()
