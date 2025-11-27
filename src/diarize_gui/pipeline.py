import os
import json
from typing import Callable, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import whisperx
from pyannote.audio import Pipeline
import requests  # NEW

from .utils import detect_device, format_timestamp

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]


class DiarizationPipelineRunner:
    """
    Encapsulates the WhisperX + diarization pipeline.
    """

    def __init__(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ):
        self.status_callback = status_callback
        self.progress_callback = progress_callback

        # store last run info for exports
        self.last_result = None
        self.last_audio_path = None
        self.last_output_dir = None
        self.last_diar_df: Optional[pd.DataFrame] = None

    def _set_status(self, text: str):
        if self.status_callback:
            self.status_callback(text)

    def _set_progress(self, value: float):
        """
        value: 0–100
        """
        if self.progress_callback:
            self.progress_callback(value)

    def _build_transcript_text(self, include_speaker: bool = True) -> str:
        """
        Build a plain-text transcript from the last_result.
        Each line: optional SPEAKER + text.
        """
        if not self.last_result or "segments" not in self.last_result:
            raise ValueError("No transcription result available for analysis.")

        lines = []
        for seg in self.last_result["segments"]:
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker", "")
            if include_speaker and speaker:
                lines.append(f"{speaker}: {text}")
            else:
                lines.append(text)
        return "\n".join(lines)


    def process_audio(
        self,
        audio_path: str,
        output_dir: str,
        model_size: str = "small",
        hf_token: Optional[str] = None,
        ):
        """
        Run transcription + alignment + diarization on the given audio file.

        Returns (txt_path, json_path) on success and stores last_result internally.
        """
        if hf_token is None or not hf_token.strip():
            raise ValueError(
                "HUGGINGFACE_TOKEN is not set. Please export it or pass it explicitly."
            )

        # remember for export functions
        self.last_audio_path = audio_path
        self.last_output_dir = output_dir
        self.last_result = None
        self.last_diar_df = None

        # ---- Device & model setup ----
        self._set_status("Detecting device...")
        self._set_progress(5)
        device = detect_device()
        compute_type = "int8" if device == "cpu" else "float16"

        self._set_status(f"Loading WhisperX model ({model_size}) on {device}...")
        self._set_progress(15)
        model = whisperx.load_model(
            model_size, device=device, compute_type=compute_type
        )

        # ---- Load audio ----
        self._set_status("Loading audio...")
        self._set_progress(25)
        audio = whisperx.load_audio(audio_path)

        # ---- Transcription ----
        self._set_status("Transcribing...")
        self._set_progress(50)
        result = model.transcribe(audio)

        # ---- Alignment ----
        self._set_status("Loading alignment model...")
        self._set_progress(60)
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )

        self._set_status("Aligning words...")
        self._set_progress(70)
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        # ---- Diarization (pyannote → DataFrame → whisperx) ----
        self._set_status("Running diarization (this may take a while)...")
        self._set_progress(85)

        diar_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=hf_token,
        )

        annotation = diar_pipeline(audio_path)

        segments = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": speaker,
                }
            )

        diarize_df = pd.DataFrame(segments)

        self._set_status("Assigning speakers to words/segments...")
        result = whisperx.assign_word_speakers(diarize_df, result)

        # store for later exports
        self.last_result = result
        self.last_diar_df = diarize_df

        # ---- Save outputs ----
        self._set_status("Saving output files...")
        self._set_progress(95)
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        txt_path = os.path.join(output_dir, f"{basename}_diarized.txt")
        json_path = os.path.join(output_dir, f"{basename}_diarized.json")

        # 1) Human-readable txt
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in result.get("segments", []):
                speaker = seg.get("speaker", "UNKNOWN")
                start = format_timestamp(seg.get("start"))
                end = format_timestamp(seg.get("end"))
                text = seg.get("text", "").strip()
                f.write(f"[{speaker} {start} - {end}] {text}\n")

        # 2) Full JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        self._set_status("Done")
        self._set_progress(100)
        return txt_path, json_path

    def analyze_with_llm(
        self,
        user_prompt: str,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """
        Call a local/remote LLM to analyze the last transcript.

        - user_prompt: the instruction you write (in Spanish/English)
        - model: optional model name (e.g. 'mixtral'); falls back to env var
        - api_url: endpoint URL; falls back to env var (e.g. Ollama)
        - api_key: optional bearer token if your server needs it

        Returns the LLM's text response.
        """
        if not user_prompt.strip():
            raise ValueError("Prompt is empty.")

        transcript = self._build_transcript_text(include_speaker=True)

        # Compose final prompt
        combined_prompt = (
            user_prompt.strip()
            + "\n\n--- TRANSCRIPCIÓN DIARIZADA ---\n"
            + transcript
        )

        # Get config from env if not provided
        api_url = api_url or os.environ.get(
            "LLM_ANALYSIS_URL", "http://localhost:11434/api/generate"
        )
        model = model or os.environ.get("LLM_ANALYSIS_MODEL", "mixtral")

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "prompt": combined_prompt,
            # Ollama-style; if you use OpenAI-style, we handle that below
            "stream": False,
        }

        self._set_status("Calling analysis model...")
        self._set_progress(50)

        resp = requests.post(api_url, json=payload, headers=headers, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        # Try to be compatible with both Ollama and OpenAI-style responses
        text = None

        # Ollama: { "response": "...", "done": true, ... }
        if isinstance(data, dict) and "response" in data:
            text = data["response"]

        # OpenAI chat/completions style
        if text is None and "choices" in data:
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception:
                try:
                    text = data["choices"][0]["text"]
                except Exception:
                    pass

        if not text:
            raise RuntimeError(f"Could not parse LLM response: {data}")

        self._set_status("Analysis done")
        self._set_progress(100)
        return text


    # ---------- Export helpers ----------
    def export_txt(self, txt_path: str):
        """
        Export a simple speaker-labeled plain text file.
        """
        if not self.last_result or "segments" not in self.last_result:
            raise ValueError("No transcription result available for TXT export.")

        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
        segments = self.last_result["segments"]

        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                speaker = seg.get("speaker", "UNKNOWN")
                start = format_timestamp(seg.get("start"))
                end = format_timestamp(seg.get("end"))
                text = seg.get("text", "").strip()
                f.write(f"[{speaker} {start}-{end}] {text}\n")

    def export_srt(self):
        if not self.has_result:
            self._show_error("Error", "No result available. Run diarization first.")
            return

        default_name = "diarized.srt"
        path = filedialog.asksaveasfilename(
            title="Save SRT file",
            defaultextension=".srt",
            filetypes=[("SRT subtitles", "*.srt"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if not path:
            return

        try:
            self.pipeline.export_srt(path)
            self._show_info("Export SRT", f"SRT saved to:\n{path}")
        except Exception as e:
            self._show_error("Error exporting SRT", str(e))

    def _srt_timestamp(self, seconds: Optional[float]) -> str:
        if seconds is None:
            seconds = 0.0
        ms = int(round(seconds * 1000))
        s = ms // 1000
        ms = ms % 1000
        h = s // 3600
        s = s % 3600
        m = s // 60
        s = s % 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def export_srt(self, srt_path: str):
        """
        Export last_result as an SRT subtitle file with speaker labels.
        """
        if not self.last_result or "segments" not in self.last_result:
            raise ValueError("No transcription result available for SRT export.")

        segments = self.last_result["segments"]
        os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)

        idx = 1
        with open(srt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                start = self._srt_timestamp(seg.get("start"))
                end = self._srt_timestamp(seg.get("end"))
                speaker = seg.get("speaker", "")
                if speaker:
                    line = f"{speaker}: {text}"
                else:
                    line = text

                f.write(f"{idx}\n{start} --> {end}\n{line}\n\n")
                idx += 1

    def export_speaker_audios(self, output_dir: str):
        """
        Export separate WAV files for each speaker by concatenating their segments.
        """
        if self.last_diar_df is None or self.last_audio_path is None:
            raise ValueError("No diarization/audio available for speaker export.")

        os.makedirs(output_dir, exist_ok=True)

        # load mono 16 kHz audio via whisperx
        audio = whisperx.load_audio(self.last_audio_path)
        sr = 16000  # whisperx.load_audio default

        basename = os.path.splitext(os.path.basename(self.last_audio_path))[0]

        for speaker, grp in self.last_diar_df.groupby("speaker"):
            chunks = []
            for _, row in grp.iterrows():
                start = int(float(row["start"]) * sr)
                end = int(float(row["end"]) * sr)
                if end > start and end <= len(audio):
                    chunks.append(audio[start:end])

            if not chunks:
                continue

            speaker_audio = np.concatenate(chunks)
            out_path = os.path.join(output_dir, f"{basename}_{speaker}.wav")
            sf.write(out_path, speaker_audio, sr)

