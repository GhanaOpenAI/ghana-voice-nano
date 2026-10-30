"""sherpa-onnx inference for Ghana Voice Nano models (Matcha-TTS acoustic model + Vocos vocoder).

A model bundle is a folder (or Hugging Face repo) holding the files written by
recipe/export_sherpa.py: acoustic .onnx files, the vocoder, tokens.txt, espeak-ng-data/
and config.json.
"""
import json
import os
import re
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

DEFAULT_REPO = "ghanaopenai/ghana-voice-nano-twi"

# Sentence-ish chunks for long text. sherpa-onnx also splits on punctuation, but chunking here
# lets callers stream audio chunk by chunk and keeps every chunk short enough to stay stable.
_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def download(repo_id=DEFAULT_REPO, revision=None, cache_dir=None):
    """Download a bundle from the Hugging Face Hub and return its local folder."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=revision, cache_dir=cache_dir)


class GhanaVoiceNano:
    """Text-to-speech for one language/voice.

    >>> tts = GhanaVoiceNano.from_pretrained("ghanaopenai/ghana-voice-nano-twi")
    >>> audio, sr = tts.synthesize("Mema wo akye. Wo ho te sɛn?")
    >>> tts.save("hello.wav", audio)
    """

    def __init__(self, model_dir, acoustic=None, num_threads=2, provider="cpu", noise_scale=None):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        variant = acoustic or self.config["default_acoustic"]
        # Accept either a variant key ("steps4") or a file name ("twi_steps4.onnx").
        self.acoustic = self.config["acoustic_models"].get(variant, variant)
        self.sample_rate = int(self.config["sample_rate"])
        d = self.model_dir
        self._tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                        acoustic_model=str(d / self.acoustic),
                        vocoder=str(d / self.config["vocoder"]),
                        lexicon="",
                        tokens=str(d / self.config["tokens"]),
                        data_dir=str(d / self.config["data_dir"]),
                        noise_scale=noise_scale if noise_scale is not None else self.config["noise_scale"],
                        length_scale=1.0,
                    ),
                    num_threads=num_threads,
                    provider=provider,
                ),
                max_num_sentences=1,
            )
        )

    @classmethod
    def from_pretrained(cls, repo_id=DEFAULT_REPO, revision=None, cache_dir=None, **kwargs):
        path = repo_id if os.path.isdir(repo_id) else download(repo_id, revision, cache_dir)
        return cls(path, **kwargs)

    @property
    def variants(self):
        return dict(self.config["acoustic_models"])

    def _generate(self, text, speed):
        audio = self._tts.generate(text, sid=0, speed=float(speed))
        return np.asarray(audio.samples, dtype=np.float32)

    def stream(self, text, speed=1.0, pause=0.25):
        """Yield audio chunk by chunk (one per sentence), for long text or low latency."""
        gap = np.zeros(int(pause * self.sample_rate), dtype=np.float32)
        chunks = [c.strip() for c in _SPLIT_RE.split(text or "") if c and c.strip()]
        for i, chunk in enumerate(chunks):
            audio = self._generate(chunk, speed)
            yield np.concatenate([audio, gap]) if i < len(chunks) - 1 else audio

    def synthesize(self, text, speed=1.0, pause=0.25):
        """Return (float32 audio in [-1, 1], sample_rate). speed > 1 speaks faster."""
        parts = list(self.stream(text, speed, pause))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return audio, self.sample_rate

    def save(self, path, audio):
        save_wav(path, audio, self.sample_rate)


def save_wav(path, audio, sample_rate):
    """Write float audio as 16-bit mono WAV with the standard library only."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def to_wav_bytes(audio, sample_rate):
    import io

    buf = io.BytesIO()
    save_wav(buf, audio, sample_rate)
    return buf.getvalue()
