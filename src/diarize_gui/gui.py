import os
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import pyperclip  # You might need to add pyclip or just let user copy manually

from .recorder import AudioRecorder
from .pipeline import DiarizationPipelineRunner
import atexit # To kill the server when app closes
from .pyannote_offline_loader import get_resource_base_path

def start_bundled_ollama():
    """
    Locates the bundled Ollama binary and starts it in server mode
    on a custom port (11435) to avoid conflicts with system Ollama.
    """
    # 1. Get absolute path to 'resources' folder
    resource_path = get_resource_base_path()
    
    # 2. Locate the binary
    ollama_bin = os.path.join(resource_path, "ollama")
    
    if not os.path.exists(ollama_bin):
        print(f"WARNING: Bundled Ollama binary not found at: {ollama_bin}")
        return None

    # 3. Ensure it is executable (permissions can be lost during packaging)
    try:
        os.chmod(ollama_bin, 0o755)
    except Exception as e:
        print(f"Warning: Could not chmod ollama binary: {e}")

    # 4. Set up environment for the subprocess
    # We store models in the user's Application Support folder
    models_dir = os.path.expanduser("~/Library/Application Support/DiarizeApp/models")
    os.makedirs(models_dir, exist_ok=True)
    
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = models_dir
    env["OLLAMA_HOST"] = "127.0.0.1:11435" # Custom port
    
    print(f"Starting bundled Ollama from {ollama_bin} on port 11435...")
    
    # 5. Launch in background
    try:
        process = subprocess.Popen(
            [ollama_bin, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return process
    except Exception as e:
        print(f"Failed to start bundled Ollama: {e}")
        return None

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

        # View history for current profile
        self.history_button = tk.Button(
            master,
            text="View lesson history",
            command=self.view_history,
        )
        self.history_button.pack(padx=10, pady=(0, 5), fill="x")

        self.audio_path = None
        self.output_dir = None
        self.is_recording = False
        self.has_result = False
        self.profile_name = None

        # Status + progress
        self.progress_var = tk.DoubleVar(value=0.0)

        self.recorder = AudioRecorder(on_status=self._set_status)
        self.pipeline = DiarizationPipelineRunner(
            status_callback=self._set_status,
            progress_callback=self._set_progress,
        )

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

        # Export buttons
        self.export_frame = tk.Frame(master)
        self.export_frame.pack(padx=10, pady=(0, 5), fill="x")

        self.analyze_button = tk.Button(
            master,
            text="Analyze transcript with LLM",
            command=self.analyze_transcript,
            state="disabled",
        )
        self.analyze_button.pack(padx=10, pady=(0, 10), fill="x")

        self.export_srt_button = tk.Button(
            self.export_frame,
            text="Export SRT",
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

        self.master.after(0, self._prompt_profile_on_startup)

        #Launch ollama
        # --- START BUNDLED OLLAMA ---
        self.ollama_process = start_bundled_ollama()
        
        # Ensure we kill the server when the GUI closes
        atexit.register(self._cleanup_ollama)
        
        # Also define a custom environment for client calls (used later in pipeline)
        # We need to tell the pipeline to talk to our custom port 11435
        os.environ["LLM_ANALYSIS_URL"] = "http://127.0.0.1:11435/api/generate"
    
    def _cleanup_ollama(self):
        if hasattr(self, 'ollama_process') and self.ollama_process:
            print("Stopping bundled Ollama server...")
            self.ollama_process.terminate()
            self.ollama_process.wait()

    def _load_existing_profiles(self):
        base_dir = os.path.expanduser("~/.whisperx_diarize_gui")
        profiles_dir = os.path.join(base_dir, "profiles")
        if not os.path.isdir(profiles_dir):
            return []
        names = []
        for name in os.listdir(profiles_dir):
            full = os.path.join(profiles_dir, name)
            if os.path.isdir(full):
                names.append(name)
        names.sort()
        return names

    def _prompt_profile_on_startup(self):
        existing = self._load_existing_profiles()

        win = tk.Toplevel(self.master)
        win.title("Select profile")
        win.geometry("400x300")
        win.transient(self.master)
        win.grab_set()

        tk.Label(
            win,
            text="Select your profile or create a new one:",
        ).pack(padx=10, pady=(10, 5), anchor="w")

        list_frame = tk.Frame(win)
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        self._profile_listbox = tk.Listbox(list_frame, height=8)
        self._profile_listbox.pack(side="left", fill="both", expand=True)

        scroll = tk.Scrollbar(list_frame, command=self._profile_listbox.yview)
        scroll.pack(side="right", fill="y")
        self._profile_listbox.configure(yscrollcommand=scroll.set)

        for name in existing:
            self._profile_listbox.insert("end", name)

        new_frame = tk.Frame(win)
        new_frame.pack(fill="x", padx=10, pady=(5, 5))

        tk.Label(new_frame, text="New profile name (optional):").pack(anchor="w")
        new_entry = tk.Entry(new_frame)
        new_entry.pack(fill="x")

        btn_frame = tk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(10, 10))

        chosen = {"name": None}

        def on_ok():
            new_name = new_entry.get().strip()
            if new_name:
                chosen["name"] = new_name
            else:
                sel = self._profile_listbox.curselection()
                if sel:
                    chosen["name"] = self._profile_listbox.get(sel[0])
                else:
                    messagebox.showerror(
                        "Profile required",
                        "Please select an existing profile or enter a new profile name.",
                        parent=win,
                    )
                    return
            win.destroy()

        def on_cancel():
            win.destroy()

        ok_btn = tk.Button(btn_frame, text="OK", command=on_ok)
        ok_btn.pack(side="right", padx=(5, 0))
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=on_cancel)
        cancel_btn.pack(side="right")

        def on_double_click(event):
            sel = self._profile_listbox.curselection()
            if not sel:
                return
            chosen["name"] = self._profile_listbox.get(sel[0])
            win.destroy()

        self._profile_listbox.bind("<Double-Button-1>", on_double_click)

        self.master.update_idletasks()
        x = self.master.winfo_rootx()
        y = self.master.winfo_rooty()
        w = self.master.winfo_width()
        h = self.master.winfo_height()
        win.update_idletasks()
        ww = win.winfo_width()
        wh = win.winfo_height()
        win.geometry(f"+{x + (w - ww) // 2}+{y + (h - wh) // 2}")
        self.master.wait_window(win)

        if chosen["name"]:
            self.profile_name = chosen["name"]
            self.profile_label.config(text=f"Profile: {self.profile_name}")

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
            "Enter profile name:",
            parent=self.master,
        )
        if name and name.strip():
            self.profile_name = name.strip()
            self.profile_label.config(text=f"Profile: {name}")

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
            self.audio_label.config(text=f"Audio file: {os.path.basename(path)}")

    def select_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir = path
            self.output_label.config(text=f"Output folder: {path}")

    def start_recording(self):
        if self.is_recording:
            return

        if not self.output_dir:
            path = filedialog.askdirectory(title="Select output folder for recording")
            if not path:
                self._show_error("Error", "Please select an output folder.")
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

    def view_history(self):
        if not self.profile_name:
            self._show_error("No profile", "No profile is set.")
            return

        base_dir = os.path.expanduser("~/.whisperx_diarize_gui")
        profile_dir = os.path.join(base_dir, "profiles", self.profile_name)
        lesson_dir = os.path.join(profile_dir, "lessons")

        if not os.path.isdir(lesson_dir):
            self._show_error("No lessons", f"No lessons found for profile '{self.profile_name}'.")
            return

        import json
        from datetime import datetime

        lessons = []
        for fname in os.listdir(lesson_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(lesson_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            ts = data.get("timestamp", "")
            audio_path = data.get("audio_path", "")
            llm_model = data.get("llm_model", "")

            try:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                ts_human = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts_human = ts

            audio_name = os.path.basename(audio_path) if audio_path else "(no audio)"
            label = f"{ts_human}  |  {audio_name}  |  {llm_model}"

            lessons.append(
                {
                    "path": path,
                    "label": label,
                    "timestamp": ts,
                    "audio_name": audio_name,
                    "llm_model": llm_model,
                    "data": data,
                }
            )

        if not lessons:
            self._show_error("No lessons", f"No valid lesson records found for profile '{self.profile_name}'.")
            return

        lessons.sort(key=lambda x: x["timestamp"], reverse=True)
        self._history_lessons = lessons

        win = tk.Toplevel(self.master)
        win.title(f"Lesson history – {self.profile_name}")
        win.geometry("700x400")

        tk.Label(win, text=f"Lessons for profile '{self.profile_name}':").pack(anchor="w", padx=10, pady=(10, 5))

        list_frame = tk.Frame(win)
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        self._history_listbox = tk.Listbox(list_frame, height=15)
        self._history_listbox.pack(side="left", fill="both", expand=True)

        scroll = tk.Scrollbar(list_frame, command=self._history_listbox.yview)
        scroll.pack(side="right", fill="y")
        self._history_listbox.configure(yscrollcommand=scroll.set)

        for lesson in lessons:
            self._history_listbox.insert("end", lesson["label"])

        btn_frame = tk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(5, 10))

        def on_open():
            sel = self._history_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            lesson = self._history_lessons[idx]
            self._open_lesson_detail(lesson)

        def on_close():
            win.destroy()

        open_btn = tk.Button(btn_frame, text="Open lesson", command=on_open)
        open_btn.pack(side="right", padx=(5, 0))
        close_btn = tk.Button(btn_frame, text="Close", command=on_close)
        close_btn.pack(side="right")

        def on_double_click(event):
            sel = self._history_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            lesson = self._history_lessons[idx]
            self._open_lesson_detail(lesson)

        self._history_listbox.bind("<Double-Button-1>", on_double_click)

    def _open_lesson_detail(self, lesson_record: dict):
        data = lesson_record.get("data", {})
        transcript_text = data.get("transcript_text", "")
        llm_response = data.get("llm_response", "")
        llm_prompt = data.get("llm_prompt", "")
        ts = data.get("timestamp", "")
        audio_path = data.get("audio_path", "")
        llm_model = data.get("llm_model", "")

        win = tk.Toplevel(self.master)
        title = f"Lesson detail – {ts}"
        if audio_path:
            title += f" – {os.path.basename(audio_path)}"
        win.title(title)
        win.geometry("1000x700")

        meta_frame = tk.Frame(win)
        meta_frame.pack(fill="x", padx=10, pady=(10, 5))

        meta_lines = []
        meta_lines.append(f"Profile: {self.profile_name}")
        if ts: meta_lines.append(f"Timestamp: {ts}")
        if audio_path: meta_lines.append(f"Audio: {audio_path}")
        if llm_model: meta_lines.append(f"LLM model: {llm_model}")

        tk.Label(meta_frame, text=" | ".join(meta_lines), wraplength=900, justify="left").pack(anchor="w")

        main_frame = tk.Frame(win)
        main_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        tk.Label(left_frame, text="Transcript:").pack(anchor="w")
        transcript_box = tk.Text(left_frame, wrap="word")
        transcript_box.pack(side="left", fill="both", expand=True)
        t_scroll = tk.Scrollbar(left_frame, command=transcript_box.yview)
        t_scroll.pack(side="right", fill="y")
        transcript_box.configure(yscrollcommand=t_scroll.set)
        transcript_box.insert("1.0", transcript_text)
        transcript_box.config(state="disabled")

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        tk.Label(right_frame, text="LLM feedback:").pack(anchor="w")
        feedback_box = tk.Text(right_frame, wrap="word")
        feedback_box.pack(side="left", fill="both", expand=True)
        f_scroll = tk.Scrollbar(right_frame, command=feedback_box.yview)
        f_scroll.pack(side="right", fill="y")
        feedback_box.configure(yscrollcommand=f_scroll.set)
        feedback_box.insert("1.0", llm_response)
        feedback_box.config(state="disabled")

        if llm_prompt:
            prompt_frame = tk.Frame(win)
            prompt_frame.pack(fill="x", padx=10, pady=(0, 10))
            tk.Label(prompt_frame, text="Prompt used:").pack(anchor="w")
            prompt_box = tk.Text(prompt_frame, wrap="word", height=5)
            prompt_box.pack(fill="x")
            prompt_box.insert("1.0", llm_prompt)
            prompt_box.config(state="disabled")

    def load_diarized_txt(self):
        path = filedialog.askopenfilename(
            title="Select diarized TXT file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.pipeline.load_segments_from_txt(path)
            self.audio_path = None
            self.output_dir = os.path.dirname(path)

            self.audio_label.config(text=f"Loaded TXT: {os.path.basename(path)}")
            self.output_label.config(text=f"Output folder: {self.output_dir}")
            self.has_result = True
            self._enable_export_buttons()
            self._set_status("Diarized TXT loaded; ready for analysis.")
        except Exception as e:
            self._show_error("Error loading TXT", str(e))

    def run_diarization(self):
        if self.is_recording:
            self._show_error("Error", "Recording in progress.")
            return

        if not self.audio_path:
            self._show_error("Error", "Select/record audio first.")
            return
        if not self.output_dir:
            self._show_error("Error", "Select output folder.")
            return

        hf_token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        if not hf_token:
            self._show_error(
                "Error",
                "HUGGINGFACE_TOKEN not set.\n\n"
                "Please set it: export HUGGINGFACE_TOKEN=\"...\"",
            )
            return

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
            self.has_result = True
            self._enable_export_buttons()
            self._show_info("Success", "Processing complete.")
        except Exception as e:
            self._set_status("Error")
            self._set_progress(0.0)
            self.has_result = False
            self._disable_export_buttons()
            self._show_error("Error", str(e))

    def _open_analysis_setup_window(self, speakers):
        win = tk.Toplevel(self.master)
        win.title("Review transcript & select student speakers")
        win.geometry("900x750")  # Slightly taller to fit everything

        # --- Top: Transcript View ---
        top_frame = tk.Frame(win)
        top_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)
        
        tk.Label(top_frame, text="Full transcript (all speakers):").pack(anchor="w")
        
        transcript_text = tk.Text(top_frame, wrap="word")
        transcript_text.pack(side="left", fill="both", expand=True)
        
        scroll1 = tk.Scrollbar(top_frame, command=transcript_text.yview)
        scroll1.pack(side="right", fill="y")
        transcript_text.configure(yscrollcommand=scroll1.set)

        try:
            full_transcript = self.pipeline.get_transcript_text(
                include_speaker=True, speaker_filters=None, max_chars=None
            )
        except Exception as e:
            full_transcript = f"<<Error building transcript: {e}>>"

        transcript_text.insert("1.0", full_transcript)
        transcript_text.config(state="disabled")

        # --- Middle: Speaker Selection & Settings ---
        mid_frame = tk.Frame(win)
        mid_frame.pack(side="top", fill="both", expand=False, padx=10, pady=5)

        # 1. Left Column: Speakers
        spk_frame = tk.Frame(mid_frame)
        spk_frame.pack(side="left", fill="y", padx=(0, 10))

        tk.Label(spk_frame, text="Select student speakers:").pack(anchor="w")
        
        # Character count label (RESTORED WARNING LOGIC)
        self._char_count_label = tk.Label(spk_frame, text="Selected student text: 0 chars", fg="black")
        self._char_count_label.pack(anchor="w", pady=(2, 5))

        self._analysis_speaker_vars = {}

        # Define update function with warning logic
        def update_char_count():
            selected = [spk for spk, var in self._analysis_speaker_vars.items() if var.get()]
            try:
                text = self.pipeline.get_transcript_text(
                    include_speaker=True,
                    speaker_filters=selected if selected else None,
                    max_chars=None,  # Get full length to check against limit
                )
            except Exception:
                text = ""
            
            length = len(text)
            max_chars = 20000  # Sync with pipeline.DEFAULT_MAX_CHARS

            if length <= max_chars:
                msg = f"Selected: {length} chars (max {max_chars}; OK)"
                self._char_count_label.config(text=msg, fg="black")
            else:
                msg = f"Selected: {length} chars (max {max_chars}; WILL BE TRUNCATED)"
                self._char_count_label.config(text=msg, fg="red")

        # Create checkboxes
        for spk in speakers:
            # Default to checking SPEAKER_00 as a guess, or unchecked
            var = tk.BooleanVar(value=(spk == "SPEAKER_00"))
            cb = tk.Checkbutton(
                spk_frame,
                text=spk,
                variable=var,
                command=update_char_count,  # update count when toggled
            )
            cb.pack(anchor="w")
            self._analysis_speaker_vars[spk] = var

        # Initialize count
        update_char_count()

        # 2. Right Column: LLM Settings & Prompt
        right_frame = tk.Frame(mid_frame)
        right_frame.pack(side="left", fill="both", expand=True)

        # LLM Settings Row (Lang + Model)
        llm_settings_frame = tk.Frame(right_frame)
        llm_settings_frame.pack(fill="x", pady=(0, 5))

        # Language Selection
        tk.Label(llm_settings_frame, text="Lang:").pack(side="left")
        self._analysis_lang_var = tk.StringVar(value="EN")  # EN or ES

        tk.Radiobutton(
            llm_settings_frame,
            text="English",
            variable=self._analysis_lang_var,
            value="EN",
        ).pack(side="left", padx=(5, 0))

        tk.Radiobutton(
            llm_settings_frame,
            text="Spanish",
            variable=self._analysis_lang_var,
            value="ES",
        ).pack(side="left", padx=(5, 15))

        # Model Selector (NEW FEATURE)
        tk.Label(llm_settings_frame, text="Ollama Model:").pack(side="left")
        self._model_var = tk.StringVar(value="mistral")
        
        suggested_models = ["mistral", "mixtral", "gemma:2b", "gemma:7b", "gemma2", "phi", "llama2", "llama3"]
        self._model_combo = ttk.Combobox(
            llm_settings_frame, 
            textvariable=self._model_var, 
            values=suggested_models,
            width=15
        )
        self._model_combo.pack(side="left", padx=(5, 0))

        # Prompt Text Area
        prompt_frame = tk.Frame(right_frame)
        prompt_frame.pack(side="top", fill="both", expand=True)

        tk.Label(prompt_frame, text="Analysis prompt (editable):").pack(anchor="w")

        default_prompt = (
            "You are an expert Spanish teacher analyzing a transcript of a 1-on-1 Spanish lesson.\n\n"
            "Your job is NOT to give a long general summary. Instead, focus mainly on the student's Spanish.\n\n"
            "Please do the following:\n"
            "1. VERY brief summary (1–2 sentences max) of what the student talked about.\n"
            "2. Identify the student's mistakes and weaknesses in Spanish (grammar, verb tenses, agreement, "
            "prepositions, pronouns, word order, vocabulary, etc.). For each important issue:\n"
            "   - Quote the original sentence or short fragment.\n"
            "   - Give a corrected or more natural version.\n"
            "   - Briefly explain why your version is better.\n"
            "3. Highlight the student's strengths.\n"
            "4. Suggest 3–5 very concrete goals for the next lessons.\n"
            "5. Provide 5–10 example sentences the student could practice.\n\n"
            "Assume that only the selected speaker IDs correspond to the student. "
            "Do NOT include the full transcript in your answer."
        )

        prompt_text = tk.Text(prompt_frame, wrap="word", height=12)
        prompt_text.pack(side="left", fill="both", expand=True)

        scroll2 = tk.Scrollbar(prompt_frame, command=prompt_text.yview)
        scroll2.pack(side="right", fill="y")
        prompt_text.configure(yscrollcommand=scroll2.set)

        prompt_text.insert("1.0", default_prompt)

        # --- Bottom: Buttons ---
        btn_frame = tk.Frame(win)
        btn_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        def on_ok():
            # Collect selected speakers
            selected = [spk for spk, var in self._analysis_speaker_vars.items() if var.get()]
            if not selected:
                messagebox.showerror("Error", "Please select at least one speaker that corresponds to the student.")
                return

            prompt = prompt_text.get("1.0", "end").strip()
            if not prompt:
                messagebox.showerror("Error", "Prompt cannot be empty.")
                return

            # --- Check Model Availability (NEW) ---
            target_model = self._model_var.get().strip()
            api_url = os.environ.get("LLM_ANALYSIS_URL", "http://localhost:11434/api/generate")
            
            # ... inside on_ok ...
            is_avail = self.pipeline.check_ollama_model_availability(target_model, api_url)
            
            if not is_avail:
                # We can't ask them to run terminal commands anymore because
                # they need to use OUR binary and OUR port/paths.
                # We should offer to download it for them right here.
                
                msg = (
                    f"The model '{target_model}' is missing from the app's internal storage.\n\n"
                    "Would you like to download it now? (This may take a few minutes)"
                )
                do_download = messagebox.askyesno("Model Missing", msg)
                
                if do_download:
                    # Run the pull command using the bundled binary and environment
                    self._run_download_thread(target_model)
                    return # Exit this function, wait for download callback
                else:
                    return

            lang = self._analysis_lang_var.get()

            # Append explicit language instruction
            if lang == "EN":
                prompt += "\n\nIMPORTANT: Write all your feedback in ENGLISH."
            else:
                prompt += "\n\nIMPORTANTE: Escribe TODA tu respuesta en ESPAÑOL."

            win.destroy()
            
            # Start analysis thread
            thread = threading.Thread(
                target=self._run_analysis_thread,
                args=(prompt, selected, target_model),
            )
            thread.daemon = True
            thread.start()

        def on_cancel():
            win.destroy()

        ok_btn = tk.Button(btn_frame, text="OK (Analyze)", command=on_ok)
        ok_btn.pack(side="right", padx=(5, 0))
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=on_cancel)
        cancel_btn.pack(side="right")

    def export_txt(self):
        if not self.has_result: return
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            try:
                self.pipeline.export_txt(path)
                self._show_info("Export", f"Saved to {path}")
            except Exception as e:
                self._show_error("Error", str(e))

    def export_srt(self):
        if not self.has_result: return
        path = filedialog.asksaveasfilename(defaultextension=".srt")
        if path:
            try:
                self.pipeline.export_srt(path)
                self._show_info("Export", f"Saved to {path}")
            except Exception as e:
                self._show_error("Error", str(e))

    def _run_download_thread(self, model_name):
        """
        Runs 'ollama pull' using the bundled binary in a separate thread,
        showing a progress window (indeterminite).
        """
        resource_path = get_resource_base_path()
        ollama_bin = os.path.join(resource_path, "ollama")
        
        # Reconstruct the environment used by the server
        models_dir = os.path.expanduser("~/Library/Application Support/DiarizeApp/models")
        env = os.environ.copy()
        env["OLLAMA_MODELS"] = models_dir
        env["OLLAMA_HOST"] = "127.0.0.1:11435"

        # Create a simple popup window
        dl_win = tk.Toplevel(self.master)
        dl_win.title(f"Downloading {model_name}...")
        dl_win.geometry("300x100")
        tk.Label(dl_win, text="Downloading model... please wait.").pack(pady=10)
        pbar = ttk.Progressbar(dl_win, mode='indeterminate')
        pbar.pack(fill='x', padx=20)
        pbar.start()

        def run_pull():
            try:
                # This blocks until download finishes
                subprocess.run(
                    [ollama_bin, "pull", model_name],
                    env=env,
                    check=True,
                    # capture output so it doesn't pop up a terminal
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE 
                )
                self.master.after(0, lambda: messagebox.showinfo("Success", f"{model_name} installed!"))
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Error", f"Download failed: {e}"))
            finally:
                self.master.after(0, dl_win.destroy)

        threading.Thread(target=run_pull, daemon=True).start()

    def analyze_transcript(self):
        if not self.has_result: return
        if not self.pipeline.last_result: return
        speakers = sorted(list(set(s.get("speaker","") for s in self.pipeline.last_result["segments"] if s.get("speaker"))))
        if not speakers:
            self._show_error("Error", "No speakers found.")
            return
        self._open_analysis_setup_window(speakers)

    def _run_analysis_thread(self, prompt: str, speakers, model: str):
        try:
            self._set_status("Running analysis...")
            self._set_progress(10)
            
            # Pass the model to the pipeline
            analysis_text = self.pipeline.analyze_with_llm(
                prompt,
                speakers=speakers,
                model=model
            )
            self._save_lesson_record(prompt, analysis_text, model)
            self._show_analysis_window(analysis_text)
        except Exception as e:
            self._set_status("Analysis error")
            self._set_progress(0.0)
            self._show_error("Error", str(e))

    def _show_analysis_window(self, text: str):
        def create_window():
            win = tk.Toplevel(self.master)
            win.title("Analysis Result")
            txt = tk.Text(win, wrap="word")
            txt.pack(fill="both", expand=True)
            txt.insert("1.0", text)
        self.master.after(0, create_window)

    def export_speaker_audio(self):
        if not self.has_result: return
        path = filedialog.askdirectory()
        if path:
            try:
                self.pipeline.export_speaker_audios(path)
                self._show_info("Export", f"Saved to {path}")
            except Exception as e:
                self._show_error("Error", str(e))

    def _set_status(self, text: str):
        self.master.after(0, lambda: self.status_label.config(text=f"Status: {text}"))

    def _set_progress(self, value: float):
        self.master.after(0, lambda: self.progress_var.set(float(value)))

    def _show_error(self, title: str, message: str):
        self.master.after(0, lambda: messagebox.showerror(title, message))

    def _show_info(self, title: str, message: str):
        self.master.after(0, lambda: messagebox.showinfo(title, message))

    def _enable_export_buttons(self):
        def enable():
            self.export_srt_button.config(state="normal")
            self.export_txt_button.config(state="normal")
            self.analyze_button.config(state="normal")
            if self.pipeline.last_diar_df is not None:
                self.export_speaker_button.config(state="normal")
        self.master.after(0, enable)

    def _disable_export_buttons(self):
        def disable():
            self.export_srt_button.config(state="disabled")
            self.export_txt_button.config(state="disabled")
            self.export_speaker_button.config(state="disabled")
            self.analyze_button.config(state="disabled")
        self.master.after(0, disable)

    def _save_lesson_record(self, prompt: str, analysis_text: str, model_used: str = "mistral"):
        import os, json
        from datetime import datetime
        if not self.profile_name: return

        base_dir = os.path.expanduser("~/.whisperx_diarize_gui")
        profile_dir = os.path.join(base_dir, "profiles", self.profile_name)
        lesson_dir = os.path.join(profile_dir, "lessons")
        os.makedirs(lesson_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lesson_{ts}.json"
        path = os.path.join(lesson_dir, filename)

        try:
            transcript_text = self.pipeline.get_transcript_text(include_speaker=True)
        except Exception:
            transcript_text = ""

        record = {
            "timestamp": ts,
            "profile": self.profile_name,
            "audio_path": self.audio_path,
            "output_dir": self.output_dir,
            "llm_model": model_used,
            "llm_prompt": prompt,
            "llm_response": analysis_text,
            "transcript_text": transcript_text,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self._show_info("Saved", f"Lesson saved to {path}")

def main():
    root = tk.Tk()
    app = DiarizationApp(root)
    root.mainloop()