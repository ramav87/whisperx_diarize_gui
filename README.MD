# WhisperX Diarization & Analysis GUI - User Guide

This application allows you to record or load audio, transcribe it using WhisperX, identify speakers (diarization), and analyze the conversation using local LLMs via Ollama.

## Prerequisites
1. **Ollama:** Must be installed and running. [Download here](https://ollama.com).
2. **HuggingFace Token:** Required for the Pyannote diarization model.
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens).
   - Create a 'Read' token.
   - You must verify you have accepted the user agreement for `pyannote/speaker-diarization-3.0`.

## Setup (First Run)
1. **Set the Token:**
   Open your terminal (or add to your `.zshrc`):
   ```bash
   export HUGGINGFACE_TOKEN="your_hf_token_here"

Usage Workflow
1. Profile Selection

On launch, select an existing profile (e.g., "Spanish_Lessons") or create a new one. This saves your lesson history separately.

2. Audio Input

Recording:

Select your microphone from the dropdown.

Click Start Recording.

When finished, click Stop Recording. The file is automatically saved and selected.

Existing File:

Click Select Existing Audio File to load .mp3, .wav, or .m4a files.

3. Run Transcription

Select Output Folder: Choose where files will be saved.

Model Size: Select small or medium for speed, or large-v2 for best accuracy.

Run: Click Run Transcription + Diarization.

Note: The first run may take time to download the Whisper models.

4. LLM Analysis

Once transcription is done, the Analyze transcript with LLM button becomes active.

Click the button.

Select Student Speakers: Check the boxes for the speakers that represent YOU (the student).

Choose Model: Select a model from the dropdown (e.g., mistral, gemma:7b).

If the model is not installed, the app will ask you to run ollama pull [model] in your terminal.

Edit Prompt: You can customize the instructions sent to the AI.

Click OK (Analyze). The result will appear in a new window and be saved to your history.

5. Review History

Click View lesson history at the top to see past transcripts and AI feedback.