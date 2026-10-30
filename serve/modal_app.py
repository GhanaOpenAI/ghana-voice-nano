"""Ghana Voice Nano TTS API on Modal (CPU, sherpa-onnx).

Deploy:   modal deploy serve/modal_app.py
Dev:      modal serve serve/modal_app.py

Endpoints (base URL printed by `modal deploy`):
  GET  /                      service info and available voices/variants
  GET  /tts?text=...          returns audio/wav  (optional: speed, variant)
  POST /tts  {"text": "...", "speed": 1.0, "variant": "steps4"}  -> audio/wav

The model is downloaded into the image at build time, so cold starts don't hit the Hub.
Set MODEL_REPO to serve another ghana-voice-nano bundle.
"""
import os

import modal

MODEL_REPO = os.environ.get("MODEL_REPO", "ghanaopenai/ghana-voice-nano-twi")
MODEL_DIR = "/model"
MAX_CHARS = 5000


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO, local_dir=MODEL_DIR, allow_patterns=["*.onnx", "*.json", "tokens.txt",
                                                                        "espeak-ng-data/**"])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sherpa-onnx>=1.12,<2", "numpy", "huggingface_hub>=0.20", "fastapi[standard]")
    .env({"MODEL_REPO": MODEL_REPO})
    .run_function(download_model)
    .add_local_python_source("ghana_voice_nano")
)

app = modal.App("ghana-voice-nano", image=image)


@app.cls(cpu=2.0, memory=2048, scaledown_window=300, timeout=600)
class TTS:
    @modal.enter()
    def load(self):
        from ghana_voice_nano import GhanaVoiceNano

        base = GhanaVoiceNano(MODEL_DIR, num_threads=2)
        self.config = base.config
        self.engines = {base.config["default_acoustic"]: base}
        for variant in base.variants:
            if variant not in self.engines:
                self.engines[variant] = GhanaVoiceNano(MODEL_DIR, acoustic=variant, num_threads=2)

    @modal.asgi_app()
    def web(self):
        import time

        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import Response
        from pydantic import BaseModel, Field

        from ghana_voice_nano import to_wav_bytes

        api = FastAPI(title="Ghana Voice Nano TTS", version="0.1.0")
        api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        class Request(BaseModel):
            text: str = Field(..., max_length=MAX_CHARS)
            speed: float = Field(1.0, ge=0.5, le=2.0)
            variant: str | None = None

        def speak(req: Request):
            text = req.text.strip()
            if not text:
                raise HTTPException(400, "text is empty")
            variant = req.variant or self.config["default_acoustic"]
            if variant not in self.engines:
                raise HTTPException(400, f"unknown variant {variant!r}; choose from {list(self.engines)}")
            t0 = time.time()
            audio, sr = self.engines[variant].synthesize(text, speed=req.speed)
            took = time.time() - t0
            return Response(
                to_wav_bytes(audio, sr),
                media_type="audio/wav",
                headers={"X-Audio-Seconds": f"{len(audio) / sr:.2f}", "X-Synthesis-Seconds": f"{took:.3f}"},
            )

        @api.get("/")
        def info():
            return {
                "model": MODEL_REPO,
                "language": self.config["language"],
                "sample_rate": self.config["sample_rate"],
                "variants": list(self.engines),
                "default_variant": self.config["default_acoustic"],
                "usage": "GET /tts?text=... or POST /tts {text, speed, variant} -> audio/wav",
            }

        @api.get("/tts")
        def tts_get(text: str, speed: float = 1.0, variant: str | None = None):
            return speak(Request(text=text, speed=speed, variant=variant))

        @api.post("/tts")
        def tts_post(req: Request):
            return speak(req)

        return api
