from __future__ import annotations

import os
import sys
import json
import re
import time
from typing import Callable, Optional, List, Any
from datetime import datetime
import numpy as np
import pandas as pd
import soundfile as sf
import requests
from .utils import detect_device, format_timestamp, is_apple_silicon
from .audio_tools import preprocess_audio_mono_16k
from .processing_backends import (
    assign_speakers_by_overlap,
    prepare_transcript_segments,
    resolve_asr_backend,
    resolve_diarization_backend,
)
from .metrics.context_adjusted import build_context_metrics

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]
TIME_PATTERN = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})")
DEFAULT_MAX_CHARS = 20000
DEFAULT_OLLAMA_ANALYSIS_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_AI_METRICS_MAX_CHARS = 8000
DEFAULT_OPENAI_AI_METRICS_MAX_CHARS = 120000

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
    Encapsulates transcription + diarization with hardware-aware backend routing.
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
        self.last_raw_result = None
        self.last_audio_path = None
        self.last_preprocessed_audio_path = None
        self.last_output_dir = None
        self.last_diar_df: Optional[pd.DataFrame] = None
        self.last_processing_meta: dict[str, Any] = {}

    @staticmethod
    def _normalize_golden_words(value) -> List[str]:
        """
        Coerce model output into a clean, unique list of up to 3 strings.
        """
        if value is None:
            return []

        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            return []

        cleaned: List[str] = []
        seen = set()
        for item in items:
            text = re.sub(r"\s+", " ", str(item)).strip(" \t\r\n-•*")
            if not text or text in seen:
                continue
            cleaned.append(text)
            seen.add(text)
            if len(cleaned) >= 3:
                break
        return cleaned

    def _set_status(self, text: str):
        if self.status_callback:
            self.status_callback(text)

    def _set_progress(self, value: float):
        """
        value: 0–100
        """
        if self.progress_callback:
            self.progress_callback(value)

    def _set_step(self, step: int, total: int, text: str, progress: Optional[float] = None):
        self._set_status(f"Step {step}/{total}: {text}")
        if progress is not None:
            self._set_progress(progress)

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
        self.last_raw_result = {"segments": segments}
        self.last_diar_df = None
        self.last_audio_path = None
        self.last_preprocessed_audio_path = None
        self.last_output_dir = os.path.dirname(txt_path)
        self.last_processing_meta = {
            "asr_backend": "txt_import",
            "diarization_backend": None,
            "word_timestamps_available": False,
            "notes": ["Loaded from existing TXT transcript."],
        }

        self._set_status("Loaded segments from TXT")
        self._set_progress(100)

    def _save_json(self, data, path):
        """
        Helper to save data to a JSON file.
        """
        import json
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved AI metrics to: {path}")
        except Exception as e:
            print(f"Error saving JSON to {path}: {e}")

    @staticmethod
    def _normalize_grammar_score(score):
        try:
            value = float(score)
        except (TypeError, ValueError):
            return score

        if 0 < value <= 10:
            value *= 10

        value = float(max(0, min(100, value)))
        return int(value) if value.is_integer() else value

    def _compute_lesson_raw_wpm(self, lesson_dir, segments=None):
        meta_path = os.path.join(lesson_dir, "meta.json")
        try:
            if segments is None:
                with open(os.path.join(lesson_dir, "segments.json"), "r", encoding="utf-8") as f:
                    segments = json.load(f) or []
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception:
            return None

        student_ids = set(meta.get("student_speakers", []))
        words = 0
        seconds = 0.0
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            spk = seg.get("speaker", "UNKNOWN")
            is_student = (spk in student_ids) or (not student_ids and "01" in str(spk))
            if not is_student:
                continue
            try:
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
            except (TypeError, ValueError):
                continue
            seconds += max(0.0, end - start)
            words += len(str(seg.get("text", "")).strip().split())
        return (words / (seconds / 60.0)) if seconds > 10 else None
            
    def compute_ai_metrics(self, lesson_dir, model=DEFAULT_OLLAMA_ANALYSIS_MODEL, mode="ollama", api_key=None):
        """
        Robustly computes metrics. 
        Attempts strict JSON parsing first, falls back to text scraping if model refuses JSON.
        """
        # --- NEW: Ensure Model Exists before we start ---
        if mode == "ollama":
            self._ensure_model_exists(model)
        # ------------------------------------------------
        import json
        import re
        
        # 1. Build Transcript
        seg_path = os.path.join(lesson_dir, "segments.json")
        transcript_path = os.path.join(lesson_dir, "transcript.txt")
        output_path = os.path.join(lesson_dir, "ai_stats.json")
        
        text_content = ""
        raw_wpm = None
        if os.path.exists(seg_path):
            try:
                with open(seg_path, 'r', encoding='utf-8') as f:
                    segs = json.load(f)
                for s in segs:
                    text_content += f"{s.get('speaker', 'Unknown')}: {s.get('text', '')}\n"
                raw_wpm = self._compute_lesson_raw_wpm(lesson_dir, segs)
            except: pass
        
        if not text_content and os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text_content = f.read()

        if not text_content: return False
        analysis_max_chars = (
            DEFAULT_OPENAI_AI_METRICS_MAX_CHARS
            if mode == "openai"
            else DEFAULT_OLLAMA_AI_METRICS_MAX_CHARS
        )

        # 2. Strict Prompt
        prompt = (
            "Analyze this language lesson. Identify the Student's mistakes.\n"
            "Respond with a strict JSON object using these keys:\n"
            "grammar_score (0-100), topics (list of 3 strings), golden_words (list of 3 complex words), corrections (int), feedback (string), "
            "topic_difficulty (number 1-5), idea_density (number 1-5), abstraction_level (number 1-10), "
            "cognitive_branching (number 1-10), technical_density (number 1-10), discourse_depth (number 1-10), "
            "lexical_retrieval_pressure (number 1-10), topic_tags (list of short strings), context_notes (short string), "
            "self_repair_observations (short string).\n\n"
            "Context rubric:\n"
            "topic_difficulty: 1=daily life/simple narration, 2=familiar concrete topic, 3=opinion or explanation, "
            "4=abstract argument, 5=technical, political, scientific, financial, philosophical, or highly abstract argument.\n"
            "idea_density: 1=simple narration with low conceptual density, 2=concrete personal topic, 3=opinion with reasons, "
            "4=abstract argument with multiple clauses, 5=dense technical/political/scientific explanation.\n"
            "abstraction_level: 1=concrete events, 5=generalized explanation, 10=epistemic/theoretical/speculative reasoning.\n"
            "cognitive_branching: 1=linear narration, 5=causal chains or comparisons, 10=nested hypotheticals/counterarguments/hedging.\n"
            "technical_density: 1=everyday vocabulary, 5=some domain vocabulary, 10=specialized technical/scientific/business terminology.\n"
            "discourse_depth: 1=short answers, 5=sustained explanation, 10=multi-step argument with evidence, tradeoffs, and synthesis.\n"
            "lexical_retrieval_pressure: 1=rehearsed familiar domain, 5=some on-the-fly searching, 10=frequent specialized concept construction.\n"
            "Do not grade the learner's intelligence or opinions. Focus only on linguistic and cognitive load.\n\n"
            "IMPORTANT FORMATTING:\n"
            "- 'golden_words' must be in the format: \"SpanishWord (EnglishTranslation)\"\n"
            "- Example: [\"desafortunadamente (unfortunately)\", \"hipótesis (hypothesis)\"]\n\n"
            "Example JSON:\n"
            "{\"grammar_score\": 75, \"topics\": [\"Food\", \"Travel\"], \"golden_words\": [\"exquisito (exquisite)\", \"viaje (journey)\"], "
            "\"corrections\": 4, \"feedback\": \"Watch your past tense.\", \"topic_difficulty\": 3, \"idea_density\": 3, "
            "\"abstraction_level\": 4, \"cognitive_branching\": 4, \"technical_density\": 2, \"discourse_depth\": 4, \"lexical_retrieval_pressure\": 3, "
            "\"topic_tags\": [\"travel\", \"food\"], \"context_notes\": \"Familiar concrete topics with some explanation.\", "
            "\"self_repair_observations\": \"Occasional restarts.\"}\n\n"
            "JSON ONLY. NO MARKDOWN."
        )

        try:
            # 3. Call LLM
            raw_response = self.analyze_with_llm(
                user_prompt=prompt,
                model=model,
                provider=mode,
                api_key=api_key,
                max_chars=analysis_max_chars,
                external_text=text_content 
            )
            
            # --- CRITICAL FIX START ---
            # Check if the LLM call actually failed before trying to parse
            if raw_response.startswith("Error:"):
                print(f"LLM Analysis Failed for {lesson_dir}: {raw_response}")
                return False  # Return False so we don't save a garbage file
            # --- CRITICAL FIX END ---
            
            # --- STRATEGY A: Try Parse JSON ---
            try:
                clean = raw_response.replace("```json", "").replace("```", "").strip()
                start = clean.find('{')
                end = clean.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = clean[start:end]
                    data = json.loads(json_str)
                    if "grammar_score" in data:
                        data["grammar_score"] = self._normalize_grammar_score(data.get("grammar_score"))
                    data["golden_words"] = self._normalize_golden_words(data.get("golden_words"))
                    data["context_metrics"] = build_context_metrics(
                        raw_grammar_score=data.get("grammar_score"),
                        raw_wpm=raw_wpm,
                        topic_difficulty=data.get("topic_difficulty"),
                        idea_density=data.get("idea_density"),
                        abstraction_level=data.get("abstraction_level"),
                        cognitive_branching=data.get("cognitive_branching"),
                        technical_density=data.get("technical_density"),
                        discourse_depth=data.get("discourse_depth"),
                        lexical_retrieval_pressure=data.get("lexical_retrieval_pressure"),
                        notes=data.get("context_notes"),
                    )
                    data["llm_provider"] = mode
                    data["llm_model"] = model
                    self._save_json(data, output_path)
                    return True
            except:
                print("JSON parsing failed, attempting text scrape...")

            # --- STRATEGY B: Scrape Text (Fallback) ---
            # ... (Rest of your fallback logic remains the same) ...
            
            fallback_data = {
                "grammar_score": 70,
                "topics": ["General Conversation"],
                "golden_words": [],
                "corrections": 0,
                "feedback": "Keep practicing!",
                "context_metrics": build_context_metrics(raw_grammar_score=70, raw_wpm=raw_wpm),
                "llm_provider": mode,
                "llm_model": model,
            }
            
            # ... (Regex matching code) ...
            
            score_match = re.search(r"Score:?\**\s*(\d+)", raw_response, re.IGNORECASE)
            if score_match: fallback_data["grammar_score"] = int(score_match.group(1))

            words_section = re.search(r"Golden Words:?(.*?)(?:\n\n|\n[A-Z])", raw_response, re.DOTALL | re.IGNORECASE)
            if words_section:
                words = re.findall(r"-\s*\*?([^\n]+)", words_section.group(1))
                if words: 
                    fallback_data["golden_words"] = self._normalize_golden_words(words)

            corr_section = re.search(r"Corrections:?(.*?)(?:\n\n|\n[A-Z])", raw_response, re.DOTALL | re.IGNORECASE)
            if corr_section:
                count = corr_section.group(1).count("\n-")
                if count > 0: fallback_data["corrections"] = count

            self._save_json(fallback_data, output_path)
            return True

        except Exception as e:
            print(f"Error computing AI metrics: {e}")
            return False

    def _ensure_model_exists(self, model_name: str):
        """
        Checks if the Ollama model exists. If not, downloads it automatically.
        """
        import subprocess
        
        # 1. Setup Paths & Env (Same as your GUI logic)
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(os.path.abspath(sys.executable))
        else:
            # Simple resource finding for dev mode
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources")
            if not os.path.exists(os.path.join(base_path, "ollama")):
                 # Fallback if resources isn't where we expect relative to pipeline.py
                 base_path = os.getcwd() 

        # Locate Binary
        ollama_bin = os.path.join(base_path, "deps", "ollama")
        if not os.path.exists(ollama_bin):
            ollama_bin = os.path.join(base_path, "ollama")
            
        if not os.path.exists(ollama_bin):
            print(f"WARNING: Could not find Ollama binary at {ollama_bin} to check for model.")
            return

        # Setup Env
        env = os.environ.copy()
        env["OLLAMA_MODELS"] = os.path.expanduser("~/Library/Application Support/DiarizeApp/models")
        env["OLLAMA_HOST"] = "127.0.0.1:11435"

        if not self._wait_for_ollama_ready(ollama_bin, env):
            print("WARNING: Ollama server is not ready yet; skipping auto-download.")
            return

        # 2. Check if model exists
        try:
            print(f"Checking if model '{model_name}' exists...")
            result = subprocess.run(
                [ollama_bin, "list"], 
                env=env, 
                capture_output=True, 
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                print(f"Failed to list Ollama models (exit {result.returncode}).")
                if stdout:
                    print(f"stdout: {stdout}")
                if stderr:
                    print(f"stderr: {stderr}")
                return
            
            if model_name not in result.stdout:
                print(f"Model '{model_name}' not found. Downloading automatically... (This may take time)")
                self._set_status(f"Downloading AI model ({model_name})...")
                
                # Run Pull
                pull_result = subprocess.run(
                    [ollama_bin, "pull", model_name], 
                    env=env, 
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if pull_result.returncode != 0:
                    stderr = (pull_result.stderr or "").strip()
                    stdout = (pull_result.stdout or "").strip()
                    print(f"Failed to download model '{model_name}' (exit {pull_result.returncode}).")
                    if stdout:
                        print(f"stdout: {stdout}")
                    if stderr:
                        print(f"stderr: {stderr}")
                    return

                print(f"Model '{model_name}' downloaded successfully.")
            else:
                print(f"Model '{model_name}' is ready.")

        except Exception as e:
            print(f"Failed to auto-download model: {e}")

    def _wait_for_ollama_ready(self, ollama_bin: str, env: dict, timeout_s: int = 30) -> bool:
        """
        Wait until the bundled Ollama server responds to `ollama list`.
        """
        import subprocess

        deadline = time.time() + timeout_s
        last_error = None
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    [ollama_bin, "list"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return True

                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                last_error = stderr or stdout or f"exit {result.returncode}"
            except Exception as e:
                last_error = str(e)

            time.sleep(0.5)

        print(f"WARNING: Ollama server at {env.get('OLLAMA_HOST')} was not ready after {timeout_s}s: {last_error}")
        return False

    def _current_runtime_flags(self) -> dict:
        return {
            "apple_silicon": is_apple_silicon(),
        }

    def _build_processing_notes(self, meta: dict) -> List[str]:
        notes = []
        for key in ("notes", "warnings", "limitations"):
            value = meta.get(key)
            if isinstance(value, list):
                notes.extend([str(v) for v in value if v])
            elif isinstance(value, str) and value:
                notes.append(value)
        return notes

    def _speaker_labels_from_result(self, result: dict) -> List[str]:
        speakers = []
        for seg in result.get("segments", []) if result else []:
            speaker = seg.get("speaker")
            if speaker and speaker not in speakers:
                speakers.append(speaker)
        return speakers

    def _segments_to_text(
        self,
        segments: List[dict],
        *,
        include_speaker: bool = True,
        highlight_low_confidence: bool = False,
    ) -> str:
        lines: List[str] = []
        for seg in segments or []:
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            speaker = seg.get("speaker", "")
            label = f"{speaker}: " if include_speaker and speaker else ""
            if highlight_low_confidence and seg.get("low_confidence"):
                label = "[LOW] " + label
            lines.append(label + text)
        return "\n".join(lines)

    def process_audio(
        self,
        audio_path: str,
        output_dir: str,
        model_size: str = "small",
        language: str = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        backend: str = "auto",
        diarization_backend: str = "pyannote",
        apple_compute_preference: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Run transcription + diarization on the given audio file.
        """

        self.last_audio_path = audio_path
        self.last_output_dir = output_dir
        self.last_result = None
        self.last_diar_df = None
        self.last_preprocessed_audio_path = None
        self.last_processing_meta = {}

        total_steps = 8

        self._set_step(1, total_steps, "Preparing audio...", 5)
        cache_dir = os.path.join(output_dir, ".cache")
        preprocess = preprocess_audio_mono_16k(audio_path, cache_dir)
        self.last_preprocessed_audio_path = preprocess.normalized_path

        self._set_step(2, total_steps, "Selecting ASR and diarization backends...", 12)
        asr_backend, asr_resolution = resolve_asr_backend(backend)
        diar_backend, diar_resolution = resolve_diarization_backend(diarization_backend)

        if batch_size is None:
            preset = (apple_compute_preference or "balanced").strip().lower()
            batch_size = {"memory_saver": 1, "balanced": 2, "quality": 4}.get(preset, 2)

        device_hint = detect_device()
        asr_config = {
            "device": device_hint,
            "compute_type": "int8" if device_hint == "cpu" else "float16",
            "batch_size": batch_size,
            "apple_compute_preference": apple_compute_preference,
            "status_callback": lambda text: self._set_step(3, total_steps, text),
            "progress_callback": self._set_progress,
        }

        self._set_step(3, total_steps, f"Preparing ASR backend ({asr_resolution['selected']})...", 15)
        if asr_resolution.get("fallback"):
            self._set_step(
                3,
                total_steps,
                f"ASR fallback active: {asr_resolution['selected']} (requested {asr_resolution['requested']})",
            )

        model_requested = model_size
        model_used = model_size
        model_fallback = None
        try:
            asr_result = asr_backend.transcribe(
                preprocess.normalized_path,
                model_size=model_requested,
                language=language,
                config=asr_config,
            )
        except Exception as first_error:
            if model_requested == "large-v3":
                model_used = "turbo"
                model_fallback = "turbo"
                self._set_step(3, total_steps, "large-v3 failed; retrying with turbo...", 18)
                try:
                    asr_result = asr_backend.transcribe(
                        preprocess.normalized_path,
                        model_size=model_used,
                        language=language,
                        config=asr_config,
                    )
                except Exception:
                    raise first_error
            else:
                raise

        self._set_step(4, total_steps, "ASR complete; starting diarization...", 84)

        diar_segments, diar_meta = diar_backend.diarize(
            preprocess.normalized_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            config={"device": "cpu" if is_apple_silicon() else device_hint},
        )
        diarize_df = pd.DataFrame(diar_segments)

        self._set_step(5, total_steps, "Cleaning transcript segments...", 88)
        cleaned_segments, raw_segments = prepare_transcript_segments(asr_result.segments)
        self._set_step(6, total_steps, "Assigning speakers to transcript...", 91)
        result_segments = assign_speakers_by_overlap(cleaned_segments, diar_segments)
        raw_segments_with_speakers = assign_speakers_by_overlap(raw_segments, diar_segments)

        raw_result = {
            "segments": raw_segments_with_speakers,
            "language": asr_result.language,
            "metadata": {
                "transcription": {
                    "backend": asr_result.backend,
                    "device": asr_result.device,
                    "compute_type": asr_result.compute_type,
                    "requested_backend": asr_resolution["requested"],
                    "selected_backend": asr_resolution["selected"],
                    "fallback_backend": asr_resolution.get("fallback"),
                    "model_requested": model_requested,
                    "model_used": model_used,
                    "model_fallback": model_fallback,
                    "word_timestamps_available": asr_result.word_timestamps_available,
                    "notes": asr_result.metadata,
                },
                "diarization": {
                    "backend": diar_meta.get("backend", diar_backend.name),
                    "device": diar_meta.get("device", "cpu"),
                    "requested_backend": diar_resolution["requested"],
                    "selected_backend": diar_resolution["selected"],
                    "fallback_backend": diar_resolution.get("fallback"),
                    "notes": diar_meta,
                },
                "audio": {
                    "source_path": audio_path,
                    "normalized_path": preprocess.normalized_path,
                    "already_normalized": preprocess.already_normalized,
                    "reused_cache": preprocess.reused_cache,
                    "sample_rate": preprocess.sample_rate,
                    "channels": preprocess.channels,
                },
            },
        }

        result = {
            "segments": result_segments,
            "language": asr_result.language,
            "metadata": raw_result["metadata"],
        }

        self.last_raw_result = raw_result
        self.last_result = result
        self.last_diar_df = diarize_df
        self.last_processing_meta = {
            "backend": asr_resolution["selected"],
            "backend_requested": asr_resolution["requested"],
            "backend_fallback": asr_resolution.get("fallback"),
            "device": asr_result.device,
            "compute_type": asr_result.compute_type,
            "model_size": model_requested,
            "model_used": model_used,
            "model_fallback": model_fallback,
            "language": language,
            "batch_size": batch_size,
            "apple_compute_preference": apple_compute_preference,
            "word_timestamps_available": asr_result.word_timestamps_available,
            "diarization_backend": diar_resolution["selected"],
            "diarization_backend_requested": diar_resolution["requested"],
            "diarization_backend_fallback": diar_resolution.get("fallback"),
            "apple_silicon": is_apple_silicon(),
            "preprocessed_audio_path": preprocess.normalized_path,
            "preprocess_reused_cache": preprocess.reused_cache,
            "preprocess_already_normalized": preprocess.already_normalized,
        }

        self._set_step(7, total_steps, "Saving output files...", 95)
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        txt_path = os.path.join(output_dir, f"{basename}_diarized.txt")
        json_path = os.path.join(output_dir, f"{basename}_diarized.json")

        raw_txt_path = os.path.join(output_dir, f"{basename}_diarized_raw.txt")
        raw_json_path = os.path.join(output_dir, f"{basename}_diarized_raw.json")
        cleaned_txt_path = os.path.join(output_dir, f"{basename}_diarized_cleaned.txt")
        artifact_path = os.path.join(output_dir, f"{basename}_transcript_artifact.json")
        highlighted_path = os.path.join(output_dir, f"{basename}_diarized_highlighted.txt")

        raw_txt = self._segments_to_text(raw_segments_with_speakers, include_speaker=True)
        cleaned_txt = self._segments_to_text(result_segments, include_speaker=True)
        highlighted_txt = self._segments_to_text(result_segments, include_speaker=True, highlight_low_confidence=True)

        with open(raw_txt_path, "w", encoding="utf-8") as f:
            f.write(raw_txt + ("\n" if raw_txt else ""))
        with open(cleaned_txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned_txt + ("\n" if cleaned_txt else ""))
        with open(highlighted_path, "w", encoding="utf-8") as f:
            f.write(highlighted_txt + ("\n" if highlighted_txt else ""))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned_txt + ("\n" if cleaned_txt else ""))

        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_result, f, ensure_ascii=False, indent=2)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "settings": self.last_processing_meta,
                    "audio": raw_result["metadata"]["audio"],
                    "raw_transcript": raw_txt,
                    "cleaned_transcript": cleaned_txt,
                    "highlighted_transcript": highlighted_txt,
                    "raw_result": raw_result,
                    "cleaned_result": result,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        self._set_step(8, total_steps, "Done", 100)
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

        # Give the bundled server a short grace period to come up.
        deadline = time.time() + 15
        last_error = None
        while time.time() < deadline:
            try:
                resp = requests.get(tags_url, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    available_models = [m.get("name", "") for m in data.get("models", [])]

                    # Simple check: exact match or match before colon
                    # e.g. "mistral" matches "mistral:latest"
                    for avail in available_models:
                        if avail == model_name:
                            return True
                        if ":" in avail and avail.split(":")[0] == model_name:
                            return True
                    return False

                last_error = f"HTTP {resp.status_code}"
            except Exception as e:
                last_error = str(e)

            time.sleep(0.5)

        try:
            print(f"WARNING: Ollama tags endpoint was not reachable: {last_error}")
        except Exception:
            # If Ollama is down or network error, assume False
            return False
        return False

    def analyze_with_llm(
        self,
        user_prompt: str,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: str = "ollama",
        speakers: Optional[List[str]] = None,
        max_chars: int = 25000,
        external_text: Optional[str] = None,
    ) -> str:
        if not user_prompt.strip():
            raise ValueError("Prompt is empty.")

        # Text Selection Logic
        if external_text:
            transcript = external_text
            if len(transcript) > max_chars:
                transcript = transcript[:max_chars] + "...(truncated)"
        else:
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
            client = OpenAIProvider(api_key=api_key, model=model or "gpt-5.4")
            self._set_status(f"Calling OpenAI ({client.model})...")
            self._set_progress(50)
            return client.analyze(combined_prompt)
            
        elif provider == "ollama":
            target_url = api_url or "http://127.0.0.1:11435/api/generate"
            target_model = model or DEFAULT_OLLAMA_ANALYSIS_MODEL

            payload = {
                "model": target_model,
                "prompt": combined_prompt,
                "stream": False,
            }

            self._set_status(f"Calling Ollama ({target_model})...")
            
            import requests
            try:
                resp = requests.post(target_url, json=payload, timeout=600)
                
                # Custom Error Handling for 404 (Model Not Found)
                if resp.status_code == 404:
                    print(f"ERROR: Ollama returned 404. It likely cannot find model '{target_model}' or the URL '{target_url}' is wrong.")
                    return f"Error: Model not found. Please run 'ollama pull {target_model}' in terminal."
                    
                resp.raise_for_status()
                data = resp.json()

                text = data.get("response")
                if not text:
                     raise RuntimeError(f"Empty response from Ollama: {data}")
                
                self._set_status("Analysis done")
                return text

            except requests.exceptions.ConnectionError:
                return "Error: Could not connect to Ollama. Is the app running? (Run 'ollama serve' in terminal)"
            except Exception as e:
                return f"Error calling Ollama: {e}"
            
        return "Error: Unknown provider"

    def load_lesson_artifacts(self, lesson_dir: str):
        """
        Restore last_result/last_diar_df/last_audio_path from a lesson folder.
        Enables export_srt/export_txt and export_speaker_audios (if audio path exists).
        Returns meta dict (may be empty).
        """
        meta_path = os.path.join(lesson_dir, "meta.json")
        seg_path = os.path.join(lesson_dir, "segments.json")
        raw_seg_path = os.path.join(lesson_dir, "segments_raw.json")
        diar_path = os.path.join(lesson_dir, "diarization.json")
        artifact_path = os.path.join(lesson_dir, "transcript_artifact.json")

        if not os.path.isfile(seg_path):
            raise FileNotFoundError(f"Missing segments.json in {lesson_dir}")

        with open(seg_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        # segments.json is a list of segments; pipeline expects {"segments": [...]}
        self.last_result = {"segments": segments}
        self.last_output_dir = lesson_dir
        self.last_raw_result = self.last_result

        if os.path.isfile(raw_seg_path):
            try:
                with open(raw_seg_path, "r", encoding="utf-8") as f:
                    raw_segments = json.load(f) or []
                self.last_raw_result = {"segments": raw_segments}
            except Exception:
                pass

        # Load meta (optional)
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        if os.path.isfile(artifact_path):
            try:
                with open(artifact_path, "r", encoding="utf-8") as f:
                    artifact = json.load(f) or {}
                if artifact.get("settings"):
                    self.last_processing_meta = artifact["settings"]
            except Exception:
                artifact = {}
        if not self.last_processing_meta:
            self.last_processing_meta = meta.get("processing", meta.get("backend_info", {})) or {
                "backend": meta.get("asr_backend"),
                "diarization_backend": meta.get("diarization_backend"),
            }

        # Option (2): use original audio path from meta, but only if it still exists
        # Support multiple historical key names to be robust:
        audio_path = (
            meta.get("source_audio_path")
            or meta.get("source_audio")
            or meta.get("last_audio_path")
            or meta.get("audio_path")
        )
        if audio_path and os.path.isfile(audio_path):
            self.last_audio_path = audio_path
        else:
            self.last_audio_path = None

        normalized_audio_path = (
            meta.get("normalized_audio_path")
            or meta.get("cached_audio_path")
            or os.path.join(lesson_dir, "normalized_audio.wav")
        )
        if normalized_audio_path and os.path.isfile(normalized_audio_path):
            self.last_preprocessed_audio_path = normalized_audio_path
        else:
            self.last_preprocessed_audio_path = None

        # diarization df optional (needed for speaker WAV export)
        if os.path.isfile(diar_path):
            with open(diar_path, "r", encoding="utf-8") as f:
                diar = json.load(f)
            self.last_diar_df = pd.DataFrame(diar)
        else:
            self.last_diar_df = None

        return meta

    # ---------- Export helpers (unchanged) ----------

    def _infer_recorded_at_iso(self, audio_path: str | None) -> str | None:
        if not audio_path:
            return None
        try:
            # Use mtime (portable). It’s seconds since epoch.
            ts = os.path.getmtime(audio_path)
            return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        except Exception:
            return None

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

        raw_segments = (self.last_raw_result or self.last_result or {}).get("segments", [])
        cleaned_segments = (self.last_result or {}).get("segments", [])

        raw_segments_clean = []
        for seg in raw_segments:
            raw_segments_clean.append(
                {
                    "start": float(seg.get("start") or 0.0),
                    "end": float(seg.get("end") or 0.0),
                    "speaker": seg.get("speaker") or "UNKNOWN",
                    "text": (seg.get("raw_text") or seg.get("text") or "").strip(),
                    "confidence": float(seg.get("confidence") or 0.0) if seg.get("confidence") is not None else None,
                    "low_confidence": bool(seg.get("low_confidence", False)),
                    "confidence_reasons": seg.get("confidence_reasons", []),
                }
            )

        segments_clean = []
        for seg in cleaned_segments:
            segments_clean.append(
                {
                    "start": float(seg.get("start") or 0.0),
                    "end": float(seg.get("end") or 0.0),
                    "speaker": seg.get("speaker") or "UNKNOWN",
                    "text": (seg.get("text") or "").strip(),
                    "raw_text": (seg.get("raw_text") or "").strip() or None,
                    "cleaned_text": (seg.get("cleaned_text") or seg.get("text") or "").strip(),
                    "confidence": float(seg.get("confidence") or 0.0) if seg.get("confidence") is not None else None,
                    "low_confidence": bool(seg.get("low_confidence", False)),
                    "confidence_reasons": seg.get("confidence_reasons", []),
                    "cleanup_applied": bool(seg.get("cleanup_applied", False)),
                }
            )

        raw_transcript = self._segments_to_text(raw_segments, include_speaker=True)
        cleaned_transcript = self._segments_to_text(cleaned_segments, include_speaker=True)
        highlighted_transcript = self._segments_to_text(
            cleaned_segments,
            include_speaker=True,
            highlight_low_confidence=True,
        )

        transcript_raw_path = os.path.join(lesson_dir, "transcript_raw.txt")
        transcript_cleaned_path = os.path.join(lesson_dir, "transcript_cleaned.txt")
        transcript_highlighted_path = os.path.join(lesson_dir, "transcript_cleaned_highlighted.txt")
        transcript_path = os.path.join(lesson_dir, "transcript.txt")
        segments_raw_path = os.path.join(lesson_dir, "segments_raw.json")
        segments_path = os.path.join(lesson_dir, "segments.json")

        with open(transcript_raw_path, "w", encoding="utf-8") as f:
            f.write(raw_transcript)
        with open(transcript_cleaned_path, "w", encoding="utf-8") as f:
            f.write(cleaned_transcript)
        with open(transcript_highlighted_path, "w", encoding="utf-8") as f:
            f.write(highlighted_transcript)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(cleaned_transcript)

        with open(segments_raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_segments_clean, f, ensure_ascii=False, indent=2)
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

        normalized_audio_path = None
        if self.last_preprocessed_audio_path and os.path.isfile(self.last_preprocessed_audio_path):
            normalized_audio_path = os.path.join(lesson_dir, "normalized_audio.wav")
            if not os.path.isfile(normalized_audio_path):
                shutil.copy2(self.last_preprocessed_audio_path, normalized_audio_path)

        # 4) meta.json
        meta = {
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "recorded_at": self._infer_recorded_at_iso(self.last_audio_path),
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
            "raw_transcript_file": "transcript_raw.txt",
            "cleaned_transcript_file": "transcript_cleaned.txt",
            "highlighted_transcript_file": "transcript_cleaned_highlighted.txt",
            "raw_segments_file": "segments_raw.json",
            "asr_backend": self.last_processing_meta.get("backend"),
            "asr_backend_requested": self.last_processing_meta.get("backend_requested"),
            "asr_backend_fallback": self.last_processing_meta.get("backend_fallback"),
            "diarization_backend": self.last_processing_meta.get("diarization_backend"),
            "diarization_backend_requested": self.last_processing_meta.get("diarization_backend_requested"),
            "diarization_backend_fallback": self.last_processing_meta.get("diarization_backend_fallback"),
            "device": self.last_processing_meta.get("device"),
            "compute_type": self.last_processing_meta.get("compute_type"),
            "batch_size": self.last_processing_meta.get("batch_size"),
            "apple_compute_preference": self.last_processing_meta.get("apple_compute_preference"),
            "word_timestamps_available": self.last_processing_meta.get("word_timestamps_available"),
            "apple_silicon": self.last_processing_meta.get("apple_silicon"),
            "preprocess_reused_cache": self.last_processing_meta.get("preprocess_reused_cache"),
            "preprocess_already_normalized": self.last_processing_meta.get("preprocess_already_normalized"),
            "normalized_audio_path": normalized_audio_path,
            "processing": self.last_processing_meta,
            "files": {
                "transcript": "transcript.txt",
                "transcript_raw": "transcript_raw.txt",
                "transcript_cleaned": "transcript_cleaned.txt",
                "segments": "segments.json",
                "segments_raw": "segments_raw.json",
                "diarization": "diarization.json" if diar_path else None,
                "normalized_audio": "normalized_audio.wav" if normalized_audio_path else None,
                "transcript_artifact": "transcript_artifact.json",
            },
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = os.path.join(lesson_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        artifact_path = os.path.join(lesson_dir, "transcript_artifact.json")
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "settings": {
                        "asr_backend": self.last_processing_meta.get("backend"),
                        "asr_backend_requested": self.last_processing_meta.get("backend_requested"),
                        "asr_backend_fallback": self.last_processing_meta.get("backend_fallback"),
                        "diarization_backend": self.last_processing_meta.get("diarization_backend"),
                        "diarization_backend_requested": self.last_processing_meta.get("diarization_backend_requested"),
                        "diarization_backend_fallback": self.last_processing_meta.get("diarization_backend_fallback"),
                        "device": self.last_processing_meta.get("device"),
                        "compute_type": self.last_processing_meta.get("compute_type"),
                        "model_size": whisper_model_size,
                        "language": language,
                        "batch_size": self.last_processing_meta.get("batch_size"),
                        "apple_compute_preference": self.last_processing_meta.get("apple_compute_preference"),
                        "apple_silicon": self.last_processing_meta.get("apple_silicon"),
                    },
                    "raw_transcript": raw_transcript,
                    "cleaned_transcript": cleaned_transcript,
                    "highlighted_transcript": highlighted_transcript,
                    "raw_segments": raw_segments_clean,
                    "cleaned_segments": segments_clean,
                    "duration_sec": self._lesson_duration_sec(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return meta

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

        audio_source = self.last_preprocessed_audio_path if self.last_preprocessed_audio_path else self.last_audio_path
        audio, sr = sf.read(audio_source, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

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
