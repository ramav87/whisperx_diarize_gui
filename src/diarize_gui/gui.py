import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk  # for Progressbar

from .recorder import AudioRecorder
from .pipeline import DiarizationPipelineRunner



class DiarizationApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("WhisperX Diarization (with Recording)")

        # Profile selection
        self.profile_frame = tk.Frame(master)
        self.profile_frame.pack(padx=10, pady=(5, 5), fill="x")

        self.profile_label = tk.Label(self.profile_frame, text="Profile: (none)")
        self.profile_label.pack(side="left")

        self.profile_button = tk.Button(
            self.profile_frame,
            text="Set profile...",
            command=self.set_profile,
        )
        self.profile_button.pack(side="right")

        self.audio_path = None
        self.output_dir = None
        self.is_recording = False
        self.has_result = False
        self.profile_name = None

        # Status + progress
        self.progress_var = tk.DoubleVar(value=0.0)

        # Recorder + pipeline (pass callbacks)
        self.recorder = AudioRecorder(on_status=self._set_status)
        self.pipeline = DiarizationPipelineRunner(
            status_callback=self._set_status,
            progress_callback=self._set_progress,
        )

        # ---------- Widgets ----------

        # Audio file label + button
        self.audio_label = tk.Label(master, text="Audio file: (none selected)")
        self.audio_label.pack(padx=10, pady=5, anchor="w")

        self.audio_button = tk.Button(
            master, text="Select Existing Audio File", command=self.select_audio
        )
        self.audio_button.pack(padx=10, pady=5, fill="x")

                # Load existing diarized TXT
        self.load_txt_button = tk.Button(
            master,
            text="Load diarized TXT for analysis",
            command=self.load_diarized_txt,
        )
        self.load_txt_button.pack(padx=10, pady=5, fill="x")

        # Output folder
        self.output_label = tk.Label(master, text="Output folder: (none selected)")
        self.output_label.pack(padx=10, pady=5, anchor="w")

        self.output_button = tk.Button(
            master, text="Select Output Folder", command=self.select_output_dir
        )
        self.output_button.pack(padx=10, pady=5, fill="x")

        # Input device selection
        self.device_label = tk.Label(master, text="Input device (microphone):")
        self.device_label.pack(padx=10, pady=(10, 0), anchor="w")

        self.device_var = tk.StringVar(value="(default)")
        self.device_menu = tk.OptionMenu(master, self.device_var, "(default)")
        self.device_menu.pack(padx=10, pady=5, fill="x")

        self.input_devices = self.recorder.list_input_devices()
        self._populate_device_menu()

        # Recording controls
        self.record_frame = tk.Frame(master)
        self.record_frame.pack(padx=10, pady=5, fill="x")

        self.start_rec_button = tk.Button(
            self.record_frame, text="Start Recording", command=self.start_recording
        )
        self.start_rec_button.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.stop_rec_button = tk.Button(
            self.record_frame,
            text="Stop Recording",
            command=self.stop_recording,
            state="disabled",
        )
        self.stop_rec_button.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # Model size
        self.model_size_label = tk.Label(master, text="Whisper model size:")
        self.model_size_label.pack(padx=10, pady=(10, 0), anchor="w")

        self.model_var = tk.StringVar(value="small")
        self.model_menu = tk.OptionMenu(
            master,
            self.model_var,
            "tiny",
            "base",
            "small",
            "medium",
            "large-v2",
        )
        self.model_menu.pack(padx=10, pady=5, fill="x")

        # condition_on_previous_text
        self.condition_var = tk.BooleanVar(value=False)
        self.condition_checkbox = tk.Checkbutton(
            master,
            text="Condition on previous text (Whisper)",
            variable=self.condition_var,
        )
        self.condition_checkbox.pack(padx=10, pady=5, anchor="w")

        # Run button
        self.run_button = tk.Button(
            master,
            text="Run Transcription + Diarization",
            command=self.run_diarization,
        )
        self.run_button.pack(padx=10, pady=10, fill="x")

        # Export buttons (disabled until we have a result)
        self.export_frame = tk.Frame(master)
        self.export_frame.pack(padx=10, pady=(0, 5), fill="x")

        # Analysis button (disabled until we have a result)
        self.analyze_button = tk.Button(
            master,
            text="Analyze transcript with LLM",
            command=self.analyze_transcript,
            state="disabled",
        )
        self.analyze_button.pack(padx=10, pady=(0, 10), fill="x")

        self.export_srt_button = tk.Button(
            self.export_frame,
            text="Export SRT (with speakers)",
            command=self.export_srt,
            state="disabled",
        )
        self.export_srt_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        self.export_txt_button = tk.Button(
            self.export_frame,
            text="Export TXT",
            command=self.export_txt,
            state="disabled",
        )
        self.export_txt_button.pack(side="left", expand=True, fill="x", padx=(5, 5))

        self.export_speaker_button = tk.Button(
            self.export_frame,
            text="Export speaker WAVs",
            command=self.export_speaker_audio,
            state="disabled",
        )
        self.export_speaker_button.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # Status label
        self.status_label = tk.Label(master, text="Status: idle")
        self.status_label.pack(padx=10, pady=(5, 2), anchor="w")

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            master,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
        )
        self.progress_bar.pack(padx=10, pady=(0, 10), fill="x")

    # ---------- Device helpers ----------

    def _populate_device_menu(self):
        menu = self.device_menu["menu"]
        menu.delete(0, "end")

        menu.add_command(
            label="(default)",
            command=lambda v="(default)": self.device_var.set(v),
        )
        self.device_var.set("(default)")

        for dev in self.input_devices:
            label = f"{dev['index']}: {dev['name']}"
            menu.add_command(
                label=label,
                command=lambda v=label: self.device_var.set(v),
            )

    def _get_selected_device_index(self):
        value = self.device_var.get()
        if value == "(default)":
            return None
        try:
            idx_str = value.split(":", 1)[0].strip()
            return int(idx_str)
        except Exception:
            return None

    def set_profile(self):
        name = simpledialog.askstring(
            "Set profile",
            "Enter profile name (e.g. 'rama', 'spanish_B2', etc.):",
            parent=self.master,
        )
        if not name:
            return

        name = name.strip()
        if not name:
            return

        self.profile_name = name
        self.profile_label.config(text=f"Profile: {name}")


    # ---------- File / folder selection ----------

    def select_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.audio_path = path
            self.audio_label.config(
                text=f"Audio file: {os.path.basename(path)}"
            )

    def select_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir = path
            self.output_label.config(text=f"Output folder: {path}")

    # ---------- Recording ----------

    def start_recording(self):
        if self.is_recording:
            return

        if not self.output_dir:
            path = filedialog.askdirectory(
                title="Select output folder for recording"
            )
            if not path:
                self._show_error(
                    "Error", "Please select an output folder before recording."
                )
                return
            self.output_dir = path
            self.output_label.config(text=f"Output folder: {path}")

        device_index = self._get_selected_device_index()

        self.recorder.start_recording(self.output_dir, device_index=device_index)
        if self.recorder.is_recording:
            self.is_recording = True
            self.start_rec_button.config(state="disabled")
            self.stop_rec_button.config(state="normal")

    def stop_recording(self):
        if not self.is_recording:
            return

        recorded_file = self.recorder.stop_recording()
        self.is_recording = False
        self.start_rec_button.config(state="normal")
        self.stop_rec_button.config(state="disabled")

        if recorded_file:
            self.audio_path = recorded_file
            self.audio_label.config(
                text=f"Audio file (recorded): {os.path.basename(recorded_file)}"
            )

    # ---------- Diarization ----------
    def load_diarized_txt(self):
        """
        Let the user pick an existing diarized TXT file and
        load it into the pipeline so that LLM analysis can run
        without processing audio.
        """
        path = filedialog.askopenfilename(
            title="Select diarized TXT file",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.pipeline.load_segments_from_txt(path)
            self.audio_path = None  # no underlying audio (optional)
            self.output_dir = os.path.dirname(path)

            # update labels so it's clear we're working from TXT
            self.audio_label.config(
                text=f"Loaded TXT: {os.path.basename(path)}"
            )
            self.output_label.config(
                text=f"Output folder: {self.output_dir}"
            )

            # we have a 'result' now (from TXT, not from pipeline)
            self.has_result = True

            # enable exports and analysis—speaker WAVs won't work without diar_df
            self._enable_export_buttons()

            # status update
            self._set_status("Diarized TXT loaded; ready for analysis.")

        except Exception as e:
            self._show_error("Error loading TXT", str(e))


    def run_diarization(self):
        if self.is_recording:
            self._show_error(
                "Error",
                "You are still recording. Please stop the recording before running diarization.",
            )
            return

        if not self.audio_path:
            self._show_error(
                "Error", "Please select or record an audio file first."
            )
            return
        if not self.output_dir:
            self._show_error(
                "Error", "Please select an output folder."
            )
            return

        hf_token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        if not hf_token:
            self._show_error(
                "Error",
                "HUGGINGFACE_TOKEN environment variable is not set.\n\n"
                "Please create a token on huggingface.co and set it, e.g.:\n"
                'export HUGGINGFACE_TOKEN="your_token_here"',
            )
            return

        # reset progress
        self._set_progress(0.0)
        self._set_status("Starting pipeline...")

        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(hf_token,),
        )
        thread.daemon = True
        thread.start()

    def _run_pipeline_thread(self, hf_token: str):
        try:
            model_size = self.model_var.get()

            self.pipeline.process_audio(
                audio_path=self.audio_path,
                output_dir=self.output_dir,
                model_size=model_size,
                hf_token=hf_token,
            )

            # mark that we have something to export
            self.has_result = True
            self._enable_export_buttons()

            self._show_info(
                "Success",
                "Transcription + diarization completed successfully.",
            )
        except Exception as e:
            self._set_status("Error")
            self._set_progress(0.0)
            self.has_result = False
            self._disable_export_buttons()
            self._show_error("Error during processing", str(e))

    def _open_analysis_setup_window(self, speakers):
        """
        Show a window where the user can:
        - Browse the full transcript
        - Select which speakers correspond to the student
        - Edit the analysis prompt
        """
        win = tk.Toplevel(self.master)
        win.title("Review transcript & select student speakers")
        win.geometry("900x600")

        # Top: transcript view
        top_frame = tk.Frame(win)
        top_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        transcript_label = tk.Label(top_frame, text="Transcripción completa (todas las voces):")
        transcript_label.pack(anchor="w")

        transcript_text = tk.Text(top_frame, wrap="word")
        transcript_text.pack(side="left", fill="both", expand=True)

        scroll1 = tk.Scrollbar(top_frame, command=transcript_text.yview)
        scroll1.pack(side="right", fill="y")
        transcript_text.configure(yscrollcommand=scroll1.set)

        # Fill transcript (all speakers, no truncation)
        try:
            full_transcript = self.pipeline.get_transcript_text(
                include_speaker=True,
                speaker_filters=None,
                max_chars=None,
            )
        except Exception as e:
            full_transcript = f"<<Error building transcript: {e}>>"

        transcript_text.insert("1.0", full_transcript)
        transcript_text.config(state="disabled")

        # Middle: speaker selection + prompt
        mid_frame = tk.Frame(win)
        mid_frame.pack(side="top", fill="both", expand=False, padx=10, pady=5)

        # Speaker checkboxes
        spk_frame = tk.Frame(mid_frame)
        spk_frame.pack(side="left", fill="y", padx=(0, 10))

        spk_label = tk.Label(spk_frame, text="Selecciona los speakers que eres TÚ (estudiante):")
        spk_label.pack(anchor="w")

        self._analysis_speaker_vars = {}
        for spk in speakers:
            var = tk.BooleanVar(value=(spk == "SPEAKER_00"))  # default guess
            cb = tk.Checkbutton(spk_frame, text=spk, variable=var)
            cb.pack(anchor="w")
            self._analysis_speaker_vars[spk] = var

        # Prompt text box
        prompt_frame = tk.Frame(mid_frame)
        prompt_frame.pack(side="left", fill="both", expand=True)

        prompt_label = tk.Label(prompt_frame, text="Prompt para el análisis (puedes editarlo):")
        prompt_label.pack(anchor="w")

        default_prompt = (
            "Eres un profesor experto de español analizando la transcripción de una clase 1-a-1.\n\n"
            "INSTRUCCIONES (muy importantes):\n"
            "1. NO hagas un resumen general largo. Como máximo, una o dos frases de resumen.\n"
            "2. El objetivo principal es analizar el español del ESTUDIANTE (no del profesor).\n"
            "3. Céntrate en:\n"
            "   - Errores gramaticales (tiempos verbales, concordancia, preposiciones, pronombres, etc.)\n"
            "   - Errores o carencias de vocabulario\n"
            "   - Frases que suenan poco naturales y cómo mejorarlas\n"
            "4. Para cada tipo de error, haz esto:\n"
            "   - Cita el fragmento original del estudiante.\n"
            "   - Propón una versión corregida o más natural.\n"
            "   - Explica brevemente por qué es mejor.\n"
            "5. Después, menciona los puntos fuertes del estudiante (qué hace bien).\n"
            "6. Termina con 3-5 objetivos concretos para la próxima clase y 5-10 frases de ejemplo para practicar.\n\n"
            "Escribe TODA la respuesta en español y no incluyas la transcripción completa (solo fragmentos cuando sea necesario).\n"
        )

        prompt_text = tk.Text(prompt_frame, wrap="word", height=12)
        prompt_text.pack(side="left", fill="both", expand=True)

        scroll2 = tk.Scrollbar(prompt_frame, command=prompt_text.yview)
        scroll2.pack(side="right", fill="y")
        prompt_text.configure(yscrollcommand=scroll2.set)

        prompt_text.insert("1.0", default_prompt)

        # Bottom: buttons
        btn_frame = tk.Frame(win)
        btn_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        def on_ok():
            # Collect selected speakers
            selected = [spk for spk, var in self._analysis_speaker_vars.items() if var.get()]
            if not selected:
                messagebox.showerror("Error", "Selecciona al menos un speaker que corresponda al estudiante.")
                return

            prompt = prompt_text.get("1.0", "end").strip()
            if not prompt:
                messagebox.showerror("Error", "El prompt no puede estar vacío.")
                return

            win.destroy()
            # Run analysis in background
            thread = threading.Thread(
                target=self._run_analysis_thread,
                args=(prompt, selected),
            )
            thread.daemon = True
            thread.start()

        def on_cancel():
            win.destroy()

        ok_btn = tk.Button(btn_frame, text="OK (analizar)", command=on_ok)
        ok_btn.pack(side="right", padx=(5, 0))
        cancel_btn = tk.Button(btn_frame, text="Cancelar", command=on_cancel)
        cancel_btn.pack(side="right")

    # ---------- Export callbacks ----------
    def export_txt(self):
        if not self.has_result:
            self._show_error("Error", "No result available. Run diarization first.")
            return

        default_name = "diarized.txt"
        path = filedialog.asksaveasfilename(
            title="Save TXT file",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if not path:
            return

        try:
            self.pipeline.export_txt(path)
            self._show_info("Export TXT", f"TXT saved to:\n{path}")
        except Exception as e:
            self._show_error("Error exporting TXT", str(e))

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


    # ---------- LLM analysis ----------
    def analyze_transcript(self):
        if not self.has_result:
            self._show_error("Error", "No result available. Run diarization first.")
            return

        # Build list of speakers from last_result
        if not self.pipeline.last_result or "segments" not in self.pipeline.last_result:
            self._show_error("Error", "No segments available for analysis.")
            return

        speakers = []
        for seg in self.pipeline.last_result["segments"]:
            spk = seg.get("speaker", "")
            if spk and spk not in speakers:
                speakers.append(spk)
        speakers.sort()

        if not speakers:
            self._show_error("Error", "No speaker labels found in transcript.")
            return

        # Create setup window
        self._open_analysis_setup_window(speakers)


    def _run_analysis_thread(self, prompt: str, speakers):
        try:
            self._set_status("Running LLM analysis...")
            self._set_progress(10)

            analysis_text = self.pipeline.analyze_with_llm(
                prompt,
                speakers=speakers,
            )

            # Save lesson under current profile (if any)
            self._save_lesson_record(prompt, analysis_text)

            # Show result in a scrollable window
            self._show_analysis_window(analysis_text)

        except Exception as e:
            self._set_status("Analysis error")
            self._set_progress(0.0)
            self._show_error("Error during analysis", str(e))



    def _show_analysis_window(self, text: str):
        def create_window():
            win = tk.Toplevel(self.master)
            win.title("LLM Analysis Result")

            txt = tk.Text(win, wrap="word")
            txt.pack(side="left", fill="both", expand=True)

            scrollbar = tk.Scrollbar(win, command=txt.yview)
            scrollbar.pack(side="right", fill="y")

            txt.configure(yscrollcommand=scrollbar.set)
            txt.insert("1.0", text)
            txt.config(state="disabled")

        self.master.after(0, create_window)

    def export_speaker_audio(self):
        if not self.has_result:
            self._show_error("Error", "No result available. Run diarization first.")
            return

        path = filedialog.askdirectory(
            title="Select folder for speaker WAV files"
        )
        if not path:
            return

        try:
            self.pipeline.export_speaker_audios(path)
            self._show_info("Export speaker WAVs", f"Speaker files saved in:\n{path}")
        except Exception as e:
            self._show_error("Error exporting speaker audio", str(e))

    def analyze_transcript(self):
        if not self.has_result:
            self._show_error("Error", "No result available. Run diarization first.")
            return

            default_prompt = (
            "Eres un profesor experto de español analizando la transcripción de una clase 1-a-1.\n\n"
            "INSTRUCCIONES (muy importantes):\n"
            "1. NO hagas un resumen general largo. Como máximo, una o dos frases de resumen.\n"
            "2. El objetivo principal es analizar el español del ESTUDIANTE (no del profesor).\n"
            "3. Céntrate en:\n"
            "   - Errores gramaticales (tiempos verbales, concordancia, preposiciones, pronombres, etc.)\n"
            "   - Errores o carencias de vocabulario\n"
            "   - Frases que suenan poco naturales y cómo mejorarlas\n"
            "4. Para cada tipo de error, haz esto:\n"
            "   - Cita el fragmento original del estudiante.\n"
            "   - Propón una versión corregida o más natural.\n"
            "   - Explica brevemente por qué es mejor.\n"
            "5. Después, menciona los puntos fuertes del estudiante (qué hace bien).\n"
            "6. Termina con 3-5 objetivos concretos para la próxima clase y 5-10 frases de ejemplo para practicar.\n\n"
            "Formato sugerido:\n"
            "A) Resumen (1-2 frases máximo)\n"
            "B) Errores y correcciones (en viñetas)\n"
            "C) Puntos fuertes\n"
            "D) Objetivos para la próxima clase\n"
            "E) Frases de ejemplo para practicar\n\n"
            "Escribe TODA la respuesta en español y no incluyas la transcripción completa (solo fragmentos cuando sea necesario).\n"
            "Asume que SPEAKER_00 es el estudiante.\n"
            )


        prompt = simpledialog.askstring(
            "LLM analysis prompt",
            "Edita el prompt si quieres, luego pulsa OK:",
            initialvalue=default_prompt,
            parent=self.master,
        )
        if not prompt:
            return

        thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(prompt,),
        )
        thread.daemon = True
        thread.start()

    # ---------- Status & progress (thread-safe) ----------

    def _set_status(self, text: str):
        # schedule update on main thread
        def update():
            self.status_label.config(text=f"Status: {text}")
        self.master.after(0, update)

    def _set_progress(self, value: float):
        # schedule update on main thread
        def update():
            self.progress_var.set(float(value))
        self.master.after(0, update)

    # ---------- Message boxes (thread-safe) ----------

    def _show_error(self, title: str, message: str):
        self.master.after(0, lambda: messagebox.showerror(title, message))

    def _show_info(self, title: str, message: str):
        self.master.after(0, lambda: messagebox.showinfo(title, message))

    def _enable_export_buttons(self):
        def enable():
            self.export_srt_button.config(state="normal")
            self.export_txt_button.config(state="normal")
            self.analyze_button.config(state="normal")

            # Only enable speaker WAVs if we have diarization dataframe (from full pipeline)
            if self.pipeline.last_diar_df is not None:
                self.export_speaker_button.config(state="normal")
            else:
                self.export_speaker_button.config(state="disabled")
        self.master.after(0, enable)

    def _disable_export_buttons(self):
        def disable():
            self.export_srt_button.config(state="disabled")
            self.export_txt_button.config(state="disabled")
            self.export_speaker_button.config(state="disabled")
            self.analyze_button.config(state="disabled")
        self.master.after(0, disable)


    def _save_lesson_record(self, prompt: str, analysis_text: str):
        """
        Save transcript + LLM feedback + metadata to a per-profile JSON file.
        """
        import os
        from datetime import datetime
        import json

        if not self.profile_name:
            # no profile -> skip saving silently, or show a gentle warning
            self._show_error(
                "No profile",
                "No profile is set. Set a profile if you want to track progress over time."
            )
            return

        # base dir: ~/.whisperx_diarize_gui/profiles/<profile>/lessons
        base_dir = os.path.expanduser("~/.whisperx_diarize_gui")
        profile_dir = os.path.join(base_dir, "profiles", self.profile_name)
        lesson_dir = os.path.join(profile_dir, "lessons")
        os.makedirs(lesson_dir, exist_ok=True)

        # timestamp-based filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lesson_{ts}.json"
        path = os.path.join(lesson_dir, filename)

        # get transcript text from pipeline
        try:
            transcript_text = self.pipeline.get_transcript_text(include_speaker=True)
        except Exception as e:
            transcript_text = f"<<Error getting transcript: {e}>>"

        # capture env model info (if any)
        import os as _os
        llm_model = _os.environ.get("LLM_ANALYSIS_MODEL", "")
        llm_url = _os.environ.get("LLM_ANALYSIS_URL", "")

        record = {
            "timestamp": ts,
            "profile": self.profile_name,
            "audio_path": self.audio_path,
            "output_dir": self.output_dir,
            "llm_model": llm_model,
            "llm_url": llm_url,
            "llm_prompt": prompt,
            "llm_response": analysis_text,
            "transcript_text": transcript_text,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # Optional: let the user know
        self._show_info("Lesson saved", f"Lesson record saved to:\n{path}")

def main():
    root = tk.Tk()
    app = DiarizationApp(root)
    root.mainloop()
