"""Command line: ghana-voice-nano --text "Mema wo akye." --out hello.wav"""
import argparse
import sys
import time

from .tts import DEFAULT_REPO, GhanaVoiceNano


def main(argv=None):
    p = argparse.ArgumentParser(description="Offline Ghanaian-language TTS (sherpa-onnx)")
    p.add_argument("--text", help="text to speak (default: read stdin)")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--model", default=DEFAULT_REPO, help="Hugging Face repo id or local bundle folder")
    p.add_argument("--variant", default=None, help="acoustic variant, e.g. steps4 (quality) or steps2 (fast)")
    p.add_argument("--speed", type=float, default=1.0, help="> 1 speaks faster")
    p.add_argument("--noise-scale", type=float, default=None)
    p.add_argument("--num-threads", type=int, default=2)
    a = p.parse_args(argv)

    text = a.text if a.text is not None else sys.stdin.read()
    tts = GhanaVoiceNano.from_pretrained(a.model, acoustic=a.variant, num_threads=a.num_threads,
                                         noise_scale=a.noise_scale)
    t0 = time.time()
    audio, sr = tts.synthesize(text, speed=a.speed)
    took = time.time() - t0
    tts.save(a.out, audio)
    secs = len(audio) / sr
    print(f"wrote {a.out}: {secs:.2f}s audio in {took:.2f}s ({secs / max(took, 1e-9):.1f}x realtime)")


if __name__ == "__main__":
    main()
