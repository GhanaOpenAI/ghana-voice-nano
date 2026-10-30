# On-device deployment (Android, iOS, Flutter, web)

Ghana Voice Nano models run fully offline through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx),
the same engine the Python package uses. No server and no network after the model is on the device.

> **Use sherpa-onnx 1.x (1.12 – 1.13.8).** sherpa-onnx 2.0 removed the espeak-ng frontend for
> Matcha models, so Matcha voices only accept a fixed lexicon. Our models read any text in the
> language through espeak-ng, so they need 1.x.

## 1. Package the model

```bash
pip install huggingface_hub
python mobile/package.py --model ghanaopenai/ghana-voice-nano-twi --variant steps4 --out dist/
```

This writes `dist/ghana-voice-nano-twi-steps4/` (~130 MB) and a `.zip` of it:

| File | What it is |
|---|---|
| `twi_steps4.onnx` | acoustic model, text to mel (`steps2` variant: same size, faster) |
| `vocos-22khz-univ.onnx` | vocoder, mel to audio |
| `tokens.txt` | phoneme to id table |
| `espeak-ng-data/` | phonemizer data for the `lfn` voice (0.7 MB) |
| `sherpa-config.json` | the file names and settings below |

## 2. Android (Kotlin)

Start from the official sample at the 1.x tag:
[`android/SherpaOnnxTts` @ v1.13.8](https://github.com/k2-fsa/sherpa-onnx/tree/v1.13.8/android/SherpaOnnxTts).
Use the prebuilt 1.13.8 `.aar` or JNI libs from the
[v1.13.8 release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.8).

1. Copy the packaged folder into `app/src/main/assets/` so you get
   `assets/ghana-voice-nano-twi-steps4/...`.
2. In `MainActivity.kt` → `initTts()`, set:

```kotlin
modelDir = "ghana-voice-nano-twi-steps4"
acousticModelName = "twi_steps4.onnx"
vocoder = "ghana-voice-nano-twi-steps4/vocos-22khz-univ.onnx"
dataDir = "ghana-voice-nano-twi-steps4/espeak-ng-data"
```

The sample copies `dataDir` from assets to app storage on first launch (espeak-ng needs real
files), then builds the config. Written out, it is:

```kotlin
val config = OfflineTtsConfig(
    model = OfflineTtsModelConfig(
        matcha = OfflineTtsMatchaModelConfig(
            acousticModel = "ghana-voice-nano-twi-steps4/twi_steps4.onnx",
            vocoder = "ghana-voice-nano-twi-steps4/vocos-22khz-univ.onnx",
            tokens = "ghana-voice-nano-twi-steps4/tokens.txt",
            dataDir = "$copiedDir/ghana-voice-nano-twi-steps4/espeak-ng-data",
            noiseScale = 0.667f,
            lengthScale = 1.0f,
        ),
        numThreads = 2,
    ),
)
val tts = OfflineTts(assetManager = assets, config = config)
val audio = tts.generate(text = "Mema wo akye. Wo ho te sɛn?", sid = 0, speed = 1.0f)
```

**App size.** 130 MB of assets is large for a Play Store APK. Common options: ship the zip as a
[Play Asset Delivery](https://developer.android.com/guide/playcore/asset-delivery) pack, or
download `<name>.zip` on first launch, unzip to app storage and pass absolute paths (then use
`OfflineTts(config = config)` without the `assetManager`).

## 3. iOS (Swift)

Sample: [`ios-swiftui/SherpaOnnxTts` @ v1.13.8](https://github.com/k2-fsa/sherpa-onnx/tree/v1.13.8/ios-swiftui/SherpaOnnxTts).
Add the packaged folder to the app bundle as a **folder reference** (blue folder) so
`espeak-ng-data/` keeps its structure, then:

```swift
func path(_ name: String) -> String {
    Bundle.main.resourcePath! + "/ghana-voice-nano-twi-steps4/" + name
}
let matcha = sherpaOnnxOfflineTtsMatchaModelConfig(
    acousticModel: path("twi_steps4.onnx"),
    vocoder: path("vocos-22khz-univ.onnx"),
    lexicon: "",
    tokens: path("tokens.txt"),
    dataDir: path("espeak-ng-data"),
    noiseScale: 0.667,
    lengthScale: 1.0
)
let modelConfig = sherpaOnnxOfflineTtsModelConfig(matcha: matcha, numThreads: 2)
var config = sherpaOnnxOfflineTtsConfig(model: modelConfig)
let tts = SherpaOnnxOfflineTtsWrapper(config: &config)
let audio = tts.generate(text: "Mema wo akye.", sid: 0, speed: 1.0)
```

## 4. Flutter / React Native / others

sherpa-onnx has [Dart/Flutter](https://pub.dev/packages/sherpa_onnx), JavaScript, C, Go, C# and
other bindings. They all take the same Matcha fields: `acousticModel`, `vocoder`, `tokens`,
`dataDir`, `lexicon=""`, `noiseScale=0.667`, `lengthScale=1.0`. Pin the binding to a 1.x
release.

## 5. Web (WebAssembly)

Build the sherpa-onnx TTS WASM app from the 1.x tree
([guide](https://k2-fsa.github.io/sherpa/onnx/tts/wasm/index.html),
[`wasm/tts` @ v1.13.8](https://github.com/k2-fsa/sherpa-onnx/tree/v1.13.8/wasm/tts)):
copy the packaged folder's files into `wasm/tts/assets/`, set the Matcha file names in the
generated config, then `./build-wasm-simd-tts.sh`. The page runs entirely in the browser.

## Speed and quality

Measured on 2 desktop CPU threads with the Twi model (realtime factor = seconds of audio per second of compute):

| Variant | Speed | Notes |
|---|---|---|
| `steps4` | ~13x realtime | default, best quality |
| `steps2` | ~16x realtime | slightly rougher; good for low-end phones |

Phones are typically 2–5x slower than a desktop core, which still leaves both variants
comfortably faster than realtime.
