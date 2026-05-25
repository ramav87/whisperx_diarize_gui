# WhisperX Diarization & Analysis GUI - User Guide

This application allows you to record or load audio, transcribe it using a hardware-aware backend, identify speakers (diarization), and analyze the conversation using local LLMs via Ollama.

## Prerequisites
(The following only applies if you are not installing via the pre-built binaries)

1. **Ollama:** Must be installed and running. [Download here](https://ollama.com).
2. **HuggingFace Token:** Required for the Pyannote diarization model when you use the standard diarization path.
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens).
   - Create a 'Read' token.
   - You must verify you have accepted the user agreement for `pyannote/speaker-diarization-3.0`.

## Setup (First Run)
1. **Set the Token:**
   Open your terminal (or add to your `.zshrc`):
   ```bash
   export HUGGINGFACE_TOKEN="your_hf_token_here"

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
- `POST /api/jobs` with multipart field `audio`
- `GET /api/jobs/{job_id}`
- `GET /api/lessons`
- `GET /api/lessons/{lesson_id}`

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

If you need a different install path, set environment variables before running
the installer:

```bash
INSTALL_DIR=/opt/diarize-server \
DATA_DIR=/srv/diarize-server \
BRANCH=optimized \
sudo bash deploy/install_linux_server.sh
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
Choose Model: Select a model from the dropdown (e.g., mistral, gemma:7b).
If the model is not installed, the app will ask you to run ollama pull [model] in your terminal.

Edit Prompt: You can customize the instructions sent to the AI.
Click OK (Analyze). The result will appear in a new window and be saved to your history.

5. **Review History**
Click View lesson history at the top to see past transcripts and AI feedback.

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

- The standard diarization path remains `pyannote`.
- In this app, pyannote is treated as a CPU stage by default on Apple Silicon.
- A future Apple-native diarization hook is exposed behind `diarization_backend = speakerkit_future`, but the current standard path still uses pyannote.

Audio is normalized to mono 16 kHz once and cached so ASR and diarization can reuse the same preprocessed WAV without repeating ffmpeg work.

Recommended Apple Silicon settings:

- `backend = auto`
- `diarization_backend = pyannote`
- `apple_compute_preference = balanced`
- `batch_size = 1` or `2` on smaller MacBooks

Notes:

- `faster-whisper` / `CTranslate2` remains the non-Mac default path and does not use MPS.
- Word timestamps are preserved when the backend supports them; otherwise the app keeps segment timestamps and records that limitation in the lesson metadata.
- Bundled pyannote diarization will prefer a `pyannote-3.1` directory if you provide one later; otherwise it falls back to the existing `pyannote` bundle.
