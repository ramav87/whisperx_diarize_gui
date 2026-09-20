# Language Learning Mobile Assistant (LLM)

Language Learning Mobile Assistant records
or imports language lessons, transcribes them, separates speakers, and builds a
longitudinal learning dashboard. The desktop dashboard also tracks learner-only
speaking time against a configurable hours-and-deadline goal. Local AI analysis
runs through Ollama; an OpenAI provider is optional.

## Supported installations

- **macOS 13+ on Apple Silicon:** recommended. `auto` uses MLX Whisper and the
  bundled FluidAudio helper when available, with WhisperX and pyannote fallbacks.
- **Windows 10/11, 64-bit:** supported from source. MLX and FluidAudio are
  macOS-only, so Windows uses WhisperX and pyannote. An NVIDIA GPU is strongly
  recommended for long recordings, but CPU processing works more slowly.
- **Linux:** supported from source, or as the processing server described below.

Python 3.10 or 3.11 is recommended. The package requires Python 3.10 or newer.

## Install the macOS app

For a pre-built release:

1. Open `Language-Learning-Mobile-Assistant-macOS.dmg`.
2. Drag `Language Learning Mobile Assistant.app` to Applications.
3. Open Language Learning Mobile Assistant and grant microphone access if you intend to record live audio.
4. If macOS blocks an ad-hoc or locally built copy, Control-click the app, choose
   **Open**, and confirm. Public releases should be Developer ID signed and notarized.

The app bundle includes Ollama and its Apple MLX runtime, FFmpeg, the offline pyannote bundle, UI assets,
and FluidAudio when it was present at build time. AI model weights are deliberately
not embedded because they are several gigabytes. By default the app stores them
on `/Volumes/XTRM-P/AI-models/DiarizeApp/models` when that drive is mounted, and
otherwise under `~/.local/share/diarize-gui/ollama-models`. Set `OLLAMA_MODELS`
before launching to choose another location.

## Run from source on macOS

Install Git, Python 3.10 or 3.11, and [FFmpeg](https://ffmpeg.org/download.html).
[Ollama](https://ollama.com/download) is optional because the app
can start a bundled binary when one is available; installing the official Ollama
app is the simplest source-development setup.

```bash
brew install git python@3.11 ffmpeg
git clone https://github.com/ramav87/whisperx_diarize_gui.git
cd whisperx_diarize_gui
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
diarize-gui
```

For Hugging Face pyannote models that are not already bundled, accept the model
terms on Hugging Face, create a read token, and export it before starting Diarize:

```bash
export HUGGINGFACE_TOKEN="your_hf_token_here"
```

## Build the macOS installation package

The build is platform-specific: create the macOS package on macOS using the same
CPU architecture you intend to distribute. Before building, provide the ignored
runtime resources `resources/pyannote/` and, optionally,
`resources/fluidaudiocli`. Install the current Ollama macOS app; the build copies
its executable and MLX runtime so supported local models work in the packaged app.
A native FFmpeg must exist either at
`resources/ffmpeg/ffmpeg` or on `PATH`.

```bash
source .venv/bin/activate
python -m pip install pyinstaller
bash build_app.sh
```

The script validates its resources, builds with `diarize.spec`, applies an ad-hoc
signature for local testing, and creates:

- `dist/Language Learning Mobile Assistant.app`
- `dist/Language-Learning-Mobile-Assistant-macOS.dmg`

For external distribution, replace the ad-hoc signature with a Developer ID
Application signature and notarize the app and disk image.

## Install from source on Windows

Install 64-bit Python 3.10 or 3.11, Git,
[FFmpeg](https://ffmpeg.org/download.html), and
[Ollama](https://ollama.com/download/windows). Ensure `ffmpeg`,
`git`, and `ollama` are available in a new PowerShell window. Ollama officially
supports Windows 10 and later.

```powershell
git clone https://github.com/ramav87/whisperx_diarize_gui.git
Set-Location whisperx_diarize_gui
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
ollama pull gemma4:e4b
diarize-gui
```

If PowerShell prevents virtual-environment activation, run this once for the
current shell and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

For an NVIDIA GPU, install the build recommended by the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/)
before `pip install -e .`, then verify that
`python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
Windows does not use MLX Whisper or FluidAudio; leave both backend selectors on
`auto` to use WhisperX and pyannote. Set `HUGGINGFACE_TOKEN` in PowerShell when
the pyannote model is not locally bundled:

```powershell
$env:HUGGINGFACE_TOKEN = "your_hf_token_here"
```

There is currently no maintained Windows installer in this repository; Windows
users should use the source installation above. A Windows package must be built
on Windows because PyInstaller does not cross-compile application bundles.

## Verify an installation

```bash
ffmpeg -version
ollama --version
python -c "import torch; print(torch.__version__)"
```

Then launch Diarize, create a profile, load a short audio file, and confirm that
the Processing status reports the selected ASR and diarization backends.

## Server mode for tablet clients

The Android tablet version should use this project as a processing server rather
than trying to run WhisperX, pyannote, and local LLMs on the tablet itself.

Start the API server on the Linux machine:

```bash
diarize-server
```

By default the server listens on `0.0.0.0:8000` and stores uploaded audio,
jobs, and lessons under `~/.local/share/diarize-server`.

Useful environment variables:

```bash
export DIARIZE_SERVER_HOST=0.0.0.0
export DIARIZE_SERVER_PORT=8000
export DIARIZE_SERVER_DATA_DIR=/srv/diarize-server
```

Initial API endpoints:

- `GET /api/health`
- `GET /api/capabilities`
- `GET /api/profiles`
- `POST /api/profiles`
- `GET /api/profiles/{profile_id}`
- `PUT /api/profiles/{profile_id}`
- `PATCH /api/profiles/{profile_id}`
- `GET /api/profiles/{profile_id}/dashboard`
- `GET /api/profiles/{profile_id}/lessons`
- `POST /api/profiles/{profile_id}/jobs` with multipart field `audio`
- `POST /api/jobs` with multipart field `audio`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/lessons`
- `GET /api/lessons/{lesson_id}`
- `GET /api/lessons/{lesson_id}/artifacts`
- `GET /api/lessons/{lesson_id}/artifacts/{artifact_name}`
- `GET /api/lessons/{lesson_id}/exports/transcript?variant=cleaned|raw|highlighted&format=txt|json`
- `GET /api/lessons/{lesson_id}/exports/analysis?format=txt|json`
- `GET /api/lessons/{lesson_id}/context`
- `PATCH /api/lessons/{lesson_id}/context`
- `PATCH /api/lessons/{lesson_id}/speakers`
- `POST /api/lessons/{lesson_id}/analysis-jobs`
- `GET /api/analysis-jobs`
- `GET /api/analysis-jobs/{job_id}`

Processing and analysis jobs are persisted under the server data directory so
clients can recover completed job status after reconnecting. Jobs that were
queued or running during a server restart are marked failed with an interruption
message rather than remaining stuck.

Profiles are first-class server records under the server data directory. New
clients should create jobs and read history through profile-scoped routes such
as `POST /api/profiles/{profile_id}/jobs` and
`GET /api/profiles/{profile_id}/lessons`. Lessons still store their `profile`
id in metadata, and `GET /api/lessons?profile=...` remains available as a
compatibility route. There is intentionally no authentication yet; future hosted
deployments should treat profiles as user-owned resources behind auth rather
than globally trusted names.

The Mac GUI defaults to server-backed processing. It uploads audio to the local
`diarize-server`, polls the API job, then mirrors the completed lesson artifacts
back into the selected local profile so existing desktop dashboard/history/export
features continue to work. The processing settings still expose `Use server API`
as an escape hatch for direct local processing when needed.
The profile picker and History view are server-first: profiles are listed and
created through the API, and history reads `GET /api/profiles/{profile_id}/lessons`
before falling back to older local profile folders.
When the GUI starts, it also starts a localhost server automatically at
`http://127.0.0.1:8000` unless one is already running. Set
`DIARIZE_GUI_START_SERVER=0` to disable this, or set `DIARIZE_GUI_SERVER_PORT`
to use another local port.

## Mobile web frontend

This repo includes an initial mobile-first web frontend in `frontend/`. It is a
static app, so it does not need a JavaScript build step.

Start the processing server:

```bash
diarize-server
```

In another terminal, serve the frontend:

```bash
python3 -m http.server 5173 --directory frontend
```

Open `http://127.0.0.1:5173/` on the server machine, or open
`http://SERVER_IP:5173/` from a phone or tablet on the same network. The UI can
connect to `http://SERVER_IP:8000`, record or upload audio, track processing
jobs, browse lessons, review speaker labels, and trigger AI analysis.

## Native Android app

The `android-app/` directory contains an initial native Android client. It uses
Android's microphone APIs to record M4A audio directly on the tablet, uploads the
recording to the existing `diarize-server` API, polls processing progress, and
can browse lessons, review speaker labels, and trigger AI analysis.

To run it:

1. Start the processing server on this machine:

   ```bash
   diarize-server
   ```

2. Open `android-app/` in Android Studio.
3. Let Android Studio sync Gradle.
4. Run the app on the tablet.
5. In the app, set the server URL to the server machine's LAN address, for
   example:

   ```text
   http://192.168.1.160:8000
   ```

The app intentionally permits cleartext HTTP for LAN development through
`network_security_config.xml`. Replace that with HTTPS before exposing the
server outside a trusted local network.

## Linux server install

For a permanent Linux host, install the app under `/opt/diarize-server` and run
it with `systemd`.

1. Clone the repo on the server and check out the branch you want to run.
2. Make sure `uv` is installed on the server.
3. Run the installer as root or with `sudo`:

```bash
sudo bash deploy/install_linux_server.sh
```

The installer will:

- create a dedicated `diarize` system user
- create `/opt/diarize-server` and `/srv/diarize-server`
- create a `uv` virtual environment
- install the package into that environment
- install and start the `diarize-server.service` unit

Helpful commands:

```bash
systemctl status diarize-server
journalctl -u diarize-server -f
sudo systemctl restart diarize-server
```

For HuggingFace pyannote diarization on the Linux server, put your token in the
optional service environment file:

```bash
sudo install -m 0600 /dev/null /etc/diarize-server.env
echo 'HUGGINGFACE_TOKEN=your_hf_token_here' | sudo tee /etc/diarize-server.env
sudo systemctl restart diarize-server
```

The server uses the offline `resources/pyannote` bundle when present. Otherwise
it falls back to `pyannote/speaker-diarization-3.1` through HuggingFace.

If you need a different install path, set environment variables before running
the installer:

```bash
INSTALL_DIR=/opt/diarize-server \
DATA_DIR=/srv/diarize-server \
BRANCH=optimized \
UV_BIN=/home/rama/local/bin/uv \
sudo bash deploy/install_linux_server.sh
```

You can also pass the `uv` binary as the first argument:

```bash
sudo bash deploy/install_linux_server.sh /home/rama/local/bin/uv
```

## Usage Workflow
1. **Profile Selection**
On launch, select an existing profile (e.g., "Spanish_Lessons") or create a new one. This saves your lesson history separately.

2. **Audio Input**
Recording:
Select your microphone from the dropdown.
Click Start Recording.
When finished, click Stop Recording. The file is automatically saved and selected.

If you have an Existing File:
Click Select Existing Audio File to load .mp3, .wav, or .m4a files.

3. **Run Transcription**
Select Output Folder: Choose where files will be saved.
Model Size: The default is now `large-v3` for best Spanish quality, with `turbo` available as the fast fallback.

Run: Click Run Transcription + Diarization.
Note: The first run may take time to download the Whisper models.

4. **LLM Analysis**
Once transcription is done, the Analyze transcript with LLM button becomes active.
Click the button.

Select Student Speakers: Check the boxes for the speakers that represent YOU (the student).
Choose Model: Select a model from the dropdown. `gemma4:12b-mlx` is recommended on Apple Silicon with 24 GB of unified memory; `gemma4:e4b` remains the faster fallback.
If the model is not installed, the app will ask you to run ollama pull [model] in your terminal.

Local Ollama analysis uses a 48K-token context and accepts up to 120,000 transcript characters. Each saved `ai_stats.json` records the provider, model, transcript coverage, learner-speaker scope, and metric schema version. Older analyses remain readable, but the Dashboard identifies them as legacy; use Backfill to regenerate comparable full-transcript metrics.

Edit Prompt: You can customize the instructions sent to the AI.
Click OK (Analyze). The result will appear in a new window and be saved to your history.

5. **Review History**
Click View lesson history at the top to see past transcripts and AI feedback.

The Dashboard's Language Growth view uses deterministic transcript measures such as MATTR lexical diversity, utterance length, and connector/subordinator density. The Vocabulary view groups frequently used content words by month. Context-adjusted values are explicitly labeled experimental and should not be read as standardized proficiency scores.

## Spanish transcript quality

The app now keeps both raw and cleaned transcript artifacts for each lesson.

- `transcript_raw.txt` and `segments_raw.json` preserve the original ASR output.
- `transcript_cleaned.txt`, `transcript.txt`, and `segments.json` contain the lightly cleaned lesson transcript used for analysis.
- `transcript_cleaned_highlighted.txt` marks low-confidence spans so you can review noisy sections quickly.
- `transcript_artifact.json` stores the structured lesson transcript bundle, including timestamps, speakers, confidence flags, raw text, cleaned text, and settings used.

The cleanup layer is intentionally conservative. It removes obvious ASR junk and repeated fragments, but it does not paraphrase learner Spanish.

Speaker review now lets you merge diarized voices by assigning multiple original speakers to the same canonical label before analysis.

## Apple Silicon optimization

On Apple Silicon Macs, the app now prefers a Mac-native ASR path by default:

1. `backend = auto` selects `MLX Whisper` first when it is available.
2. If MLX Whisper is not installed or cannot load a model, the app falls back to `openai-whisper` on MPS when available.
3. If neither Apple-specific path is available, the app falls back to the existing WhisperX path.

The diarization stage stays separate from ASR.

- `diarization_backend = auto` selects FluidAudio on Apple Silicon when its bundled helper is available.
- FluidAudio uses its offline Core ML pipeline and requests Apple Neural Engine compute.
- Pyannote remains the cross-platform fallback and runs as a CPU stage on Apple Silicon in this app.
- `single_speaker` skips speaker separation when speaker labels are unnecessary.

Audio is normalized to mono 16 kHz once and cached so ASR and diarization can reuse the same preprocessed WAV without repeating ffmpeg work.

Recommended Apple Silicon settings:

- `backend = auto`
- `diarization_backend = auto`
- `apple_compute_preference = balanced`
- `batch_size = 1` or `2` on smaller MacBooks

Notes:

- `faster-whisper` / `CTranslate2` remains the non-Mac default path and does not use MPS.
- The first FluidAudio run downloads and compiles its Core ML models; later runs reuse the local model cache.
- Word timestamps are preserved when the backend supports them; otherwise the app keeps segment timestamps and records that limitation in the lesson metadata.
- Bundled pyannote diarization will prefer a `pyannote-3.1` directory if you provide one later; otherwise it falls back to the existing `pyannote` bundle.

## Compare Whisper vs Gemma transcription

Use the comparison script to run the app's existing ASR path against a chunked
Gemma audio transcription path on the same lesson audio:

```bash
python -m diarize_gui.compare_transcription_pipelines /path/to/lesson.wav \
  --existing-backend auto \
  --whisper-model small \
  --gemma-model google/gemma-3n-E4B-it \
  --gemma-device mps \
  --output-dir /tmp/lesson-transcription-comparison
```

The Gemma path splits the normalized lesson audio into 30 second WAV chunks,
transcribes each chunk with an audio-capable local/Hugging Face Gemma checkpoint,
then writes side-by-side transcripts, segment JSON, timing, realtime factors, and
rough text agreement metrics to `transcription_comparison.json`.

By default, the comparison runner starts each transcription backend in a separate
Python process. This avoids native library collisions between MLX, Torch, and
other model runtimes on macOS. Use `--in-process` only when debugging the script
itself.

The Gemma runner is optional and imports `transformers` only when that path runs.
If you only want to time the existing app ASR path first, add `--skip-gemma`.
If you only want a low-risk Gemma smoke test, skip Whisper and run one short
chunk first:

```bash
python -m diarize_gui.compare_transcription_pipelines /path/to/lesson.wav \
  --skip-existing \
  --gemma-model google/gemma-3n-E2B-it \
  --gemma-device cpu \
  --gemma-chunk-seconds 10 \
  --gemma-max-chunks 1 \
  --gemma-max-new-tokens 96 \
  --output-dir /tmp/lesson-gemma-smoke
```

To compare both ASR paths on the same short slice before processing a full
lesson, use `--clip-start-seconds` and `--clip-max-seconds`:

```bash
python -m diarize_gui.compare_transcription_pipelines /path/to/lesson.wav \
  --existing-backend auto \
  --whisper-model small \
  --gemma-model google/gemma-3n-E2B-it \
  --gemma-local-files-only \
  --gemma-device cpu \
  --clip-start-seconds 0 \
  --clip-max-seconds 30 \
  --gemma-chunk-seconds 10 \
  --gemma-max-new-tokens 96 \
  --output-dir /tmp/lesson-whisper-vs-gemma-30s
```
