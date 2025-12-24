import os
import json
import re
from typing import Callable, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd
import soundfile as sf
import whisperx
from pyannote.audio import Pipeline
import requests
from .utils import detect_device, format_timestamp
from .pyannote_offline_loader import load_offline_pipeline  # <--- ADD THIS

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]
TIME_PATTERN = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})")
DEFAULT_MAX_CHARS = 20000

def parse_time_to_seconds(t: str) -> float:
    """
    Parse 'HH:MM:SS.mmm' into seconds as float.
    """
    m = TIME_PATTERN.match(t.strip())
    if not m:
        return 0.0
    h = int(m.group("h"))
    m_ = int(m.group("m"))
    s = int(m.group("s"))
    ms = int(m.group("ms"))
    return h * 3600 + m_ * 60 + s + ms / 1000.0


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

    def load_segments_from_txt(self, txt_path: str):
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"TXT file not found: {txt_path}")

        segments = []
        pattern = re.compile(
            r"^\[(?P<speaker>\S+)\s+(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-\s*"
            r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(?P<text>.*)$"
        )

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                m = pattern.match(line)
                if not m:
                    continue
                speaker = m.group("speaker")
                start_s = parse_time_to_seconds(m.group("start"))
                end_s = parse_time_to_seconds(m.group("end"))
                text = m.group("text")

                segments.append(
                    {
                        "start": float(start_s),
                        "end": float(end_s),
                        "speaker": speaker,
                        "text": text,
                    }
                )

        if not segments:
            raise ValueError("No segments could be parsed from TXT file.")

        self.last_result = {"segments": segments}
        self.last_diar_df = None
        self.last_audio_path = None
        self.last_output_dir = os.path.dirname(txt_path)

        self._set_status("Loaded segments from TXT")
        self._set_progress(100)

    def process_audio(
        self,
        audio_path: str,
        output_dir: str,
        model_size: str = "small",
        language: str = None
       
        ):
        """
        Run transcription + alignment + diarization on the given audio file.
        """
        
        # REMOVED: The check for hf_token
        
        self.last_audio_path = audio_path
        self.last_output_dir = output_dir
        self.last_result = None
        self.last_diar_df = None

        self._set_status("Detecting device...")
        self._set_progress(5)
        device = detect_device()
        compute_type = "int8" if device == "cpu" else "float16"

        self._set_status(f"Loading WhisperX model ({model_size}) on {device}...")
        self._set_progress(15)
        model = whisperx.load_model(
            model_size, device=device, compute_type=compute_type
        )

        self._set_status("Loading audio...")
        self._set_progress(25)
        audio = whisperx.load_audio(audio_path)

        self._set_status("Transcribing...")
        self._set_progress(50)
        result = model.transcribe(audio, language=language, task="transcribe")

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

        # --- CHANGED: Use Offline Loader ---
        self._set_status("Running diarization (offline model)...")
        self._set_progress(85)

        # Use the helper we wrote to load local files
        try:
            diar_pipeline = load_offline_pipeline()
        except Exception as e:
            raise RuntimeError(f"Failed to load offline Pyannote model: {e}")

        # Run inference
        annotation = diar_pipeline(audio_path)
        # -----------------------------------

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

        self.last_result = result
        self.last_diar_df = diarize_df

        self._set_status("Saving output files...")
        self._set_progress(95)
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        txt_path = os.path.join(output_dir, f"{basename}_diarized.txt")
        json_path = os.path.join(output_dir, f"{basename}_diarized.json")

        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in result.get("segments", []):
                speaker = seg.get("speaker", "UNKNOWN")
                start = format_timestamp(seg.get("start"))
                end = format_timestamp(seg.get("end"))
                text = seg.get("text", "").strip()
                f.write(f"[{speaker} {start} - {end}] {text}\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        self._set_status("Done")
        self._set_progress(100)
        return txt_path, json_path

    def get_transcript_text(
        self,
        include_speaker: bool = True,
        speaker_filters: Optional[List[str]] = None,
        max_chars: Optional[int] = None,
        ) -> str:
        if not self.last_result or "segments" not in self.last_result:
            raise ValueError("No transcription result available.")

        allowed = set(speaker_filters) if speaker_filters else None

        lines = []
        for seg in self.last_result["segments"]:
            text = seg.get("text", "").strip()
            if not text:
                continue

            speaker = seg.get("speaker", "")

            if allowed is not None and speaker not in allowed:
                continue

            if include_speaker and speaker:
                lines.append(f"{speaker}: {text}")
            else:
                lines.append(text)

        transcript = "\n".join(lines)

        if max_chars is not None and len(transcript) > max_chars:
            transcript = transcript[-max_chars:]

        return transcript
    
    # --- NEW: Check model availability ---
    def check_ollama_model_availability(self, model_name: str, api_url: str) -> bool:
        """
        Returns True if model_name is found in local Ollama tags.
        Returns False otherwise (or if Ollama is unreachable).
        """
        # Usually api_url is http://localhost:11434/api/generate
        # We need the tags endpoint: http://localhost:11434/api/tags
        base_url = api_url.rsplit("/api/", 1)[0]
        tags_url = f"{base_url}/api/tags"

        try:
            resp = requests.get(tags_url, timeout=3)
            if resp.status_code != 200:
                return False
            data = resp.json()
            # data['models'] is a list of dicts: [{'name': 'mistral:latest', ...}, ...]
            available_models = [m.get("name", "") for m in data.get("models", [])]
            
            # Simple check: exact match or match before colon
            # e.g. "mistral" matches "mistral:latest"
            for avail in available_models:
                if avail == model_name:
                    return True
                if ":" in avail and avail.split(":")[0] == model_name:
                    return True
            return False
        except Exception:
            # If Ollama is down or network error, assume False
            return False

    def analyze_with_llm(
        self,
        user_prompt: str,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: str = "ollama",
        speakers: Optional[List[str]] = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        if not user_prompt.strip():
            raise ValueError("Prompt is empty.")

        transcript = self.get_transcript_text(
            include_speaker=True,
            speaker_filters=speakers,
            max_chars=max_chars,
        )

        speakers_str = ", ".join(speakers) if speakers else "TODOS"
        combined_prompt = (
            user_prompt.strip()
            + "\n\n--- FILTERED TRANSCRIPT (speakers: "
            + speakers_str
            + ") ---\n"
            + transcript
        )

        if provider == "openai":

            from .openai_provider import OpenAIProvider
            client = OpenAIProvider(api_key=api_key, model=model or "gpt-5.1")

            self._set_status(f"Calling OpenAI ({client.model})...")
            self._set_progress(50)
            text = client.analyze(combined_prompt)
            self._set_status("Analysis done")
            self._set_progress(100)

            return text
        elif provider == "ollama":
            api_url = api_url or os.environ.get(
                "LLM_ANALYSIS_URL", "http://localhost:11434/api/generate"
            )
            
            # Override env var if model is passed
            model = model or os.environ.get("LLM_ANALYSIS_MODEL", "mistral")

            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": model,
                "prompt": combined_prompt,
                "stream": False,
            }

            self._set_status(f"Calling analysis model ({model})...")
            self._set_progress(50)

            import requests
            resp = requests.post(api_url, json=payload, headers=headers, timeout=600)
            resp.raise_for_status()
            data = resp.json()

            text = None
            if isinstance(data, dict) and "response" in data:
                text = data["response"]
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

    def load_lesson_artifacts(self, lesson_dir: str):
        # segments
        seg_path = os.path.join(lesson_dir, "segments.json")
        with open(seg_path, "r", encoding="utf-8") as f:
            self.last_result = json.load(f)

        # diarization (if separate)
        diar_path = os.path.join(lesson_dir, "diarization.json")
        if os.path.isfile(diar_path):
            with open(diar_path, "r", encoding="utf-8") as f:
                diar = json.load(f)
            self.last_diar = diar  # or build a dataframe your exporter expects

        # meta
        meta_path = os.path.join(lesson_dir, "meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}

        # audio path resolution (prefer embedded audio)
        embedded_audio = os.path.join(lesson_dir, meta.get("saved_audio_filename", "audio.wav"))
        if os.path.isfile(embedded_audio):
            self.last_audio_path = embedded_audio
        else:
            p = meta.get("source_audio_path")
            self.last_audio_path = p if p and os.path.isfile(p) else None

        return meta

    # ---------- Export helpers (unchanged) ----------
    def _lesson_duration_sec(self) -> float:
        """Best-effort duration from segments/diarization."""
        ends = []
        if self.last_result and "segments" in self.last_result:
            ends.extend([float(s.get("end") or 0.0) for s in self.last_result["segments"]])
        if self.last_diar_df is not None and not self.last_diar_df.empty:
            ends.extend([float(x) for x in self.last_diar_df["end"].tolist()])
        return float(max(ends)) if ends else 0.0


    def save_lesson_artifacts(
        self,
        lesson_dir: str,
        *,
        profile_name: Optional[str] = None,
        whisper_model_size: Optional[str] = None,
        language: Optional[str] = None,
        contextual: Optional[bool] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        extra_meta: Optional[dict] = None,
        ) -> dict:
        """
        Persist everything needed to re-open a lesson and (optionally) re-export speaker WAVs
        without re-running transcription/diarization.

        Writes:
        - transcript.txt
        - segments.json  (start/end/speaker/text)
        - diarization.json (start/end/speaker) if available
        - meta.json

        Returns meta dict.
        """
        if not self.last_result or "segments" not in self.last_result:
            raise ValueError("No transcription segments available to save.")

        os.makedirs(lesson_dir, exist_ok=True)

        # 1) transcript.txt (human-readable)
        transcript_path = os.path.join(lesson_dir, "transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(self.get_transcript_text(include_speaker=True))

        # 2) segments.json (canonical for future features)
        segments_clean = []
        for seg in self.last_result["segments"]:
            segments_clean.append(
                {
                    "start": float(seg.get("start") or 0.0),
                    "end": float(seg.get("end") or 0.0),
                    "speaker": seg.get("speaker") or "UNKNOWN",
                    "text": (seg.get("text") or "").strip(),
                }
            )

        segments_path = os.path.join(lesson_dir, "segments.json")
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments_clean, f, ensure_ascii=False, indent=2)

        # 3) diarization-only segments (optional but important for speaker WAV export)
        diar_path = None
        if self.last_diar_df is not None and not self.last_diar_df.empty:
            diar_clean = []
            for _, row in self.last_diar_df.iterrows():
                diar_clean.append(
                    {
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "speaker": str(row["speaker"]),
                    }
                )
            diar_path = os.path.join(lesson_dir, "diarization.json")
            with open(diar_path, "w", encoding="utf-8") as f:
                json.dump(diar_clean, f, ensure_ascii=False, indent=2)
        
        import shutil

        if self.last_audio_path and os.path.isfile(self.last_audio_path):
            dst = os.path.join(lesson_dir, "audio.wav")
            if not os.path.isfile(dst):
                shutil.copy2(self.last_audio_path, dst)

        # 4) meta.json
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "profile": profile_name,
            "source_audio_path": self.last_audio_path,   # critical for future speaker WAV export
            "saved_audio_filename": "audio.wav",  # if you copy it
            "output_dir": self.last_output_dir,
            "duration_sec": self._lesson_duration_sec(),
            "num_segments": len(segments_clean),
            "num_speakers": len({s["speaker"] for s in segments_clean}),
            "whisper_model_size": whisper_model_size,
            "language": language,
            "contextual": contextual,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "files": {
                "transcript": "transcript.txt",
                "segments": "segments.json",
                "diarization": "diarization.json" if diar_path else None,
            },
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = os.path.join(lesson_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return meta

    def load_lesson_artifacts(self, lesson_dir: str):
        """
        Restore last_result/last_diar_df/last_audio_path from a lesson folder.
        Enables export_srt/export_txt and export_speaker_audios (if audio path exists).
        """
        meta_path = os.path.join(lesson_dir, "meta.json")
        seg_path = os.path.join(lesson_dir, "segments.json")
        diar_path = os.path.join(lesson_dir, "diarization.json")

        if not os.path.isfile(seg_path):
            raise FileNotFoundError(f"Missing segments.json in {lesson_dir}")

        with open(seg_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        self.last_result = {"segments": segments}
        self.last_output_dir = lesson_dir

        # optional meta
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.last_audio_path = meta.get("source_audio")
        else:
            self.last_audio_path = None

        # diarization df optional
        if os.path.isfile(diar_path):
            with open(diar_path, "r", encoding="utf-8") as f:
                diar = json.load(f)
            self.last_diar_df = pd.DataFrame(diar)
        else:
            self.last_diar_df = None


    def export_txt(self, txt_path: str):
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
        if self.last_diar_df is None or self.last_audio_path is None:
            raise ValueError("No diarization/audio available for speaker export.")

        os.makedirs(output_dir, exist_ok=True)

        audio = whisperx.load_audio(self.last_audio_path)
        sr = 16000

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