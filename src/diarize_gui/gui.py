import os
import sys
import threading
import subprocess
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
import stat
import math
from .dashboard import DashboardFrame

# Import backend logic
from .utils import obfuscate_secret, deobfuscate_secret, estimate_openai_cost
from .recorder import AudioRecorder
from .pipeline import DiarizationPipelineRunner
from .pyannote_offline_loader import get_resource_base_path

# --- CONFIGURATION ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

LANGUAGE_MAP = {
    "Auto-Detect": None, "English": "en", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Portuguese": "pt", "Russian": "ru",
    "Chinese": "zh", "Japanese": "ja", "Korean": "ko", "Dutch": "nl",
    "Polish": "pl", "Turkish": "tr", "Hindi": "hi", "Arabic": "ar",
    "Czech": "cs", "Greek": "el", "Hebrew": "he", "Hungarian": "hu",
    "Indonesian": "id", "Malay": "ms", "Romanian": "ro", "Swedish": "sv",
    "Ukrainian": "uk", "Vietnamese": "vi"
}

def start_bundled_ollama():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.abspath(sys.executable))
    else:
        base_path = get_resource_base_path()

    # Look in deps (macOS bundle) or dev path
    ollama_bin = os.path.join(base_path, "deps", "ollama")
    if not os.path.exists(ollama_bin):
        # Dev fallback
        ollama_bin = os.path.join(get_resource_base_path(), "ollama")
    
    if not os.path.exists(ollama_bin):
        print(f"CRITICAL ERROR: Bundled Ollama binary not found at: {ollama_bin}")
        return None

    try:
        os.chmod(ollama_bin, 0o755)
    except Exception:
        pass

    models_dir = os.path.expanduser("~/Library/Application Support/DiarizeApp/models")
    os.makedirs(models_dir, exist_ok=True)
    
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = models_dir
    env["OLLAMA_HOST"] = "127.0.0.1:11435"
    
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

class ToolTip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self.after_id = None

        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _=None):
        self.after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tipwindow:
            return

        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() - 10

        self.tipwindow = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)

        label = ctk.CTkLabel(
            tw,
            text=self.text,
            fg_color="#2B2B2B",
            text_color="#E6E6E6",
            corner_radius=6,
            padx=8,
            pady=4,
            justify="left",
        )
        label.pack()

    def _hide(self, _=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None



class DiarizationApp:
    def __init__(self, master):
        self.master = master
        self.profile_config = {}
        self.current_lesson_dir = None
        self._assign_window_open = False

        master.title("Lesson Recording and Analysis App")
        master.geometry("750x900")

        # --- LOGIC INIT ---
        self.audio_path = None
        self.output_dir = None
        self.is_recording = False
        self.has_result = False
        self.profile_name = None
        
        self.recorder = AudioRecorder(on_status=self._set_status)
        self.pipeline = DiarizationPipelineRunner(
            status_callback=self._set_status,
            progress_callback=self._set_progress,
        )

        # --- ICONS ---
        self._load_icons()

        # --- UI LAYOUT ---
        self.tab_view = ctk.CTkTabview(self.master)
        self.master.configure(fg_color="#1E1E1E")

        self.tab_view = ctk.CTkTabview(self.master, fg_color="#1E1E1E")
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_dash = self.tab_view.add("Dashboard")
        self.tab_dash.configure(fg_color="#1E1E1E")

        self.tab_studio = self.tab_view.add("Studio")
        self.tab_studio.configure(fg_color="#1E1E1E")
        
        # Set "Studio" as the parent for all existing UI elements
        # We will pass self.tab_studio to _build_ui instead of self.master
        self._build_ui(parent=self.tab_studio)
        
        # Initialize Dashboard (empty until profile loads)
        self.dashboard = None

        # --- STARTUP ---
        # Defer profile prompt slightly so UI renders first
        self.master.after(200, self._prompt_profile_on_startup)
        
        # Start Ollama
        self.ollama_process = start_bundled_ollama()
        import atexit
        atexit.register(self._cleanup_ollama)
        os.environ["LLM_ANALYSIS_URL"] = "http://127.0.0.1:11435/api/generate"

    def _load_icons(self):
        def load_icon(name):
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.abspath(sys.executable))
                # Check common bundled locations
                possible = [
                    os.path.join(base_path, "resources", "icons", name),
                    os.path.join(base_path, "icons", name),
                    os.path.join(base_path, "..", "Resources", "icons", name)
                ]
            else:
                # Source mode: src/diarize_gui/gui_modern.py -> ... -> resources/icons
                curr = os.path.dirname(os.path.abspath(__file__))
                root = os.path.abspath(os.path.join(curr, "..", ".."))
                possible = [os.path.join(root, "resources", "icons", name)]

            for p in possible:
                if os.path.exists(p):
                    return ctk.CTkImage(light_image=Image.open(p), dark_image=Image.open(p), size=(20, 20))
            return None

        self.icon_user = load_icon("user.png")
        self.icon_mic = load_icon("mic.png")
        self.icon_folder = load_icon("folder.png")

    def _build_ui(self, parent):
        # 1. PROFILE HEADER
        self.profile_frame = ctk.CTkFrame(parent, corner_radius=10, fg_color = "#262626")
        self.profile_frame.pack(padx=15, pady=(15, 5), fill="x")

        self.profile_label = ctk.CTkLabel(
            self.profile_frame, 
            text="Profile: (none)", 
            font=("Roboto", 16, "bold"),
            image=self.icon_user,
            compound="left",
            padx=10
        )
        self.profile_label.pack(side="left", padx=10, pady=10)

        self.profile_btn = ctk.CTkButton(self.profile_frame, text="Change", width=80, command=self.set_profile)
        self.profile_btn.pack(side="right", padx=(5, 10))
        
        self.history_btn = ctk.CTkButton(
            self.profile_frame, 
            text="History", 
            width=80, 
            fg_color="transparent", 
            border_width=2, 
            command=self.view_history
        )
        self.history_btn.pack(side="right", padx=0)

        # 2. INPUT CARD
        self.input_card = ctk.CTkFrame(parent, fg_color = "#262626")
        self.input_card.pack(padx=15, pady=5, fill="x")
        
        ctk.CTkLabel(self.input_card, text="Input Source", font=("Roboto", 14, "bold")).pack(anchor="w", padx=15, pady=(10,5))

        # A. File Select
        self.file_row = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.file_row.pack(fill="x", padx=10, pady=5)
        
        self.audio_btn = ctk.CTkButton(self.file_row, text="Select Audio", image=self.icon_folder, command=self.select_audio, width=120)
        self.audio_btn.pack(side="left", padx=5)
        
        self.audio_label = ctk.CTkLabel(self.file_row, text="(No file selected)", text_color="gray")
        self.audio_label.pack(side="left", padx=5)

        # Load TXT Button (at bottom)
        self.load_txt_btn = ctk.CTkButton(self.file_row, text="Load Existing TXT", fg_color="transparent", border_width=1, text_color=("gray10", "gray90"), command=self.load_diarized_txt)
        self.load_txt_btn.pack(padx=5)

        ctk.CTkLabel(self.input_card, text="- OR -", text_color="gray", font=("Arial", 10)).pack()

        # B. Recording
        self.rec_row = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.rec_row.pack(fill="x", padx=10, pady=5)

        # Device list
        self.input_devices = self.recorder.list_input_devices()
        dev_names = ["(default)"] + [f"{d['index']}: {d['name']}" for d in self.input_devices]
        
        self.device_var = ctk.StringVar(value="(default)")
        self.device_menu = ctk.CTkOptionMenu(self.rec_row, variable=self.device_var, values=dev_names, width=220, height = 40)
        self.device_menu.pack(side="left", padx=5)

        self.start_rec_btn = ctk.CTkButton(
            self.rec_row, text="REC", width=120, height = 40,font=("Roboto", 14, "bold"), 
            fg_color="#c0392b", hover_color="#b71c1c", 
            image=self.icon_mic, command=self.start_recording
        )
        self.start_rec_btn.pack(side="left", padx=5)
        
        self.stop_rec_btn = ctk.CTkButton(self.rec_row, text="STOP", width=120,height = 40,
        font=("Roboto", 14, "bold"), state="disabled", command=self.stop_recording)
        self.stop_rec_btn.pack(side="left", padx=5)

        # Mic level meter (initially hidden/disabled)
        self.mic_level_label = ctk.CTkLabel(self.rec_row, text="Mic:", text_color="gray")
        self.mic_level_bar = ctk.CTkProgressBar(self.rec_row, width=120)
        self.mic_level_bar.set(0.0)
        self.mic_level_db = ctk.CTkLabel(self.rec_row, text="", text_color="gray")

        self.mic_level_label.pack(side="left", padx=(12, 6))
        self.mic_level_bar.pack(side="left", padx=(0, 6))
        self.mic_level_db.pack(side="left", padx=(0, 0))

        # 3. SETTINGS CARD
        self.settings_card = ctk.CTkFrame(parent, fg_color = "#262626")
        self.settings_card.pack(padx=15, pady=10, fill="x")
        
        ctk.CTkLabel(self.settings_card, text="Processing Settings", font=("Roboto", 14, "bold")).pack(anchor="w", padx=15, pady=(10,5))
        
        grid = ctk.CTkFrame(self.settings_card, fg_color="transparent")
        grid.pack(fill="x", padx=10, pady=5)

        # Output folder
        self.out_btn = ctk.CTkButton(grid, text="Output Folder",font=("Roboto", 18, "bold"), width=120, height = 40, command=self.select_output_dir)
        self.out_btn.grid(row=0, column=0, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(grid, text="(None)", text_color="gray")
        self.output_label.grid(row=0, column=1, padx=5, sticky="w")

        lbl = ctk.CTkLabel(grid, text="Expected speakers:")
        lbl.grid(row=0, column=1, padx=5, sticky="e")
        ToolTip(
        lbl,
        "Set the expected number of speakers for diarization.\n"
        "• Auto: WhisperX decides (may over-split).\n"
        "• 2 is recommended for tutor/student lessons."
        )
        lbl.configure(cursor="question_arrow")

        self.exp_spk_var = ctk.StringVar(value="2")
        self.exp_spk_menu = ctk.CTkOptionMenu(
            grid,
            variable=self.exp_spk_var,
            values=["Auto", "1", "2", "3", "4", "5", "6"],
            width=100
        )
        self.exp_spk_menu.grid(row=0, column=2, padx=5, sticky="w")

        ToolTip(
            self.exp_spk_menu,
            "For group sessions, use Auto.\n"
            "For lessons, 2 is usually best."
        )

        # Model Size
        ctk.CTkLabel(grid, text="Transcription Model Size:").grid(row=1, column=0, padx=(0, 8), pady=5, sticky="w")
        self.model_var = ctk.StringVar(value="small")
        self.model_menu = ctk.CTkOptionMenu(
            grid, variable=self.model_var,
            values=["tiny", "base", "small", "medium", "large-v2"],
            width=110
        )
        self.model_menu.grid(row=1, column=1, padx=(0, 20), pady=5, sticky="w")

        # Language
        ctk.CTkLabel(grid, text="Language:").grid(row=1, column=2, padx=(0, 8), pady=5, sticky="w")
        self.lang_var = ctk.StringVar(value="Auto-Detect")
        self.lang_combo = ctk.CTkOptionMenu(
            grid, variable=self.lang_var,
            values=list(LANGUAGE_MAP.keys()),
            width=140
        )
        self.lang_combo.grid(row=1, column=3, padx=(0, 0), pady=5, sticky="w")
        
        # Checkbox
        self.context_var = ctk.BooleanVar(value=False)
        self.context_cb = ctk.CTkCheckBox(grid, text="Context", variable=self.context_var)
        self.context_cb.grid(row=1, column=4, padx=(15, 0), sticky="w")

        # 4. ACTION CARD (Run + Progress + Status)
        self.action_card = ctk.CTkFrame(parent, fg_color = "#262626")
        self.action_card.pack(padx=15, pady=8, fill="x")

        # Run button (hero)
        self.run_btn = ctk.CTkButton(
            self.action_card,
            text="RUN PROCESSING",
            height=50,
            font=("Roboto", 18, "bold"),
            command=self.run_diarization
        )
        self.run_btn.pack(padx=12, pady=(12, 8), fill="x")

        self.assign_btn = ctk.CTkButton(self.action_card, text="Assign Speakers…",width=160,
        command=self.open_assign_speakers,state = "disabled")
        self.assign_btn.pack(side="left", padx=8, pady=8)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.action_card)
        self.progress_bar.pack(padx=12, pady=(0, 6), fill="x")
        self.progress_bar.set(0)

        # Status label (more visible)
        self.status_label = ctk.CTkLabel(
            self.action_card,
            text="Status: Idle",
            text_color="gray",
            font=("Roboto", 12)
        )
        self.status_label.pack(padx=12, pady=(0, 12), anchor="w")

        # 5. POST-PROCESSING CARD
        self.post_card = ctk.CTkFrame(parent)
        self.post_card.pack(padx=15, pady=5, fill="x")

        # Analyze card
        self.analyze_card = ctk.CTkFrame(parent)
        self.analyze_card.pack(padx=15, pady=(6, 10), fill="x")

        self.analyze_btn = ctk.CTkButton(
            self.analyze_card,
            text="Analyze with AI Assistant",
            height=46,
            fg_color="#2e7d32",        # green
            hover_color="#256628",     # darker green on hover
            text_color="white",
            font=("Roboto", 16, "bold"),
            command=self.analyze_transcript
        )
        self.analyze_btn.pack(padx=12, pady=(10, 4), fill="x")

        self.analyze_help = ctk.CTkLabel(
            self.analyze_card,
            text="",
            text_color="gray",
            font=("Roboto", 12),
            anchor="w",
            justify="left"
        )
        self.analyze_help.pack(padx=12, pady=(0, 8), anchor="w")

        exp_row = ctk.CTkFrame(self.post_card, fg_color="transparent")
        exp_row.pack(fill="x", padx=10, pady=(0,10))
        
        self.export_srt_btn = ctk.CTkButton(exp_row, text="Export SRT", state="disabled", width=80, command=self.export_srt)
        self.export_srt_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        self.export_txt_btn = ctk.CTkButton(exp_row, text="Export TXT", state="disabled", width=80, command=self.export_txt)
        self.export_txt_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        self.export_wav_btn = ctk.CTkButton(exp_row, text="Export Speakers", state="disabled", width=80, command=self.export_speaker_audio)
        self.export_wav_btn.pack(side="left", padx=5, expand=True, fill="x")
        self.export_help = ctk.CTkLabel(
            exp_row,
            text="",
            text_color="gray",
            font=("Roboto", 12),
            anchor="w",
            justify="left"
        )
        self._update_analyze_ui_state()

    def open_assign_speakers(self):
        if not getattr(self, "current_lesson_dir", None):
            messagebox.showerror("Error", "No lesson loaded. Run processing or load a transcript first.")
            return

        seg_path = os.path.join(self.current_lesson_dir, "segments.json")
        if not os.path.isfile(seg_path):
            messagebox.showerror("Error", "This lesson has no segments.json (nothing to assign).")
            return

        try:
            with open(seg_path, "r", encoding="utf-8") as f:
                segs = json.load(f) or []
            speakers = sorted({s.get("speaker") for s in segs if s.get("speaker")})
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read segments.json:\n{e}")
            return

        if not speakers:
            messagebox.showinfo("Assign Speakers", "No speaker labels found in segments.")
            return

        self._open_assign_speakers_window(speakers)

    # --- LOGIC METHODS ---
    def _update_analyze_ui_state(self):
        """
        Updates Analyze button + helper text based on whether a transcript exists and whether we're busy.
        """
        busy = getattr(self, "_is_processing", False) or getattr(self, "_is_analyzing", False)

        has_transcript = False
        try:
            txt = self.pipeline.get_transcript_text(include_speaker=True, max_chars=100)
            has_transcript = bool(txt and txt.strip())
        except Exception:
            has_transcript = False

        # --- defaults: prevent stale UI ---
        self.analyze_btn.configure(state="disabled")
        self.analyze_help.configure(text="")
        try:
            self.export_wav_btn.configure(state="disabled")
            self.export_help.configure(text="")
        except Exception:
            pass

        # Assign button availability
        can_assign = False
        lesson_dir = getattr(self, "current_lesson_dir", None)
        if lesson_dir:
            seg_path = os.path.join(lesson_dir, "segments.json")
            can_assign = os.path.isfile(seg_path)
        try:
            self.assign_btn.configure(state="normal" if can_assign else "disabled")
        except Exception:
            pass

        if busy:
            self.analyze_help.configure(text="Working… please wait.")
            try:
                self.export_help.configure(text="Working… please wait.")
            except Exception:
                pass
            return

        if not has_transcript:
            self.analyze_help.configure(text="To enable analysis: run processing or load an existing TXT transcript.")
            try:
                self.export_help.configure(text="To enable export: run processing or load an existing TXT transcript.")
            except Exception:
                pass
            return

        # If we have transcript, we're at least eligible for analysis once assignment is done
        # Now enforce speaker assignment if segments exist
        if lesson_dir and os.path.isfile(os.path.join(lesson_dir, "segments.json")):
            if not self._has_speaker_assignment(lesson_dir):
                self.analyze_help.configure(text="Action required: Assign Student Speaker(s) before analysis.")
                try:
                    self.export_help.configure(text="Action required: Assign Student Speaker(s) before Speaker WAV export.")
                except Exception:
                    pass
                return

        self.analyze_btn.configure(state="normal")
        self.analyze_help.configure(text="Ready for analysis.")

        try:
            self.export_wav_btn.configure(state="normal")
            self.export_help.configure(text="Ready: Speaker WAV export available.")
            self._set_status("Ready for Export and Analysis")
        except Exception:
            pass


    # --- PROFILE CONFIG (OpenAI key, provider prefs) ---
    def _profile_lessons_dir(self):
        d = self._profile_dir()
        if not d:
            return None
        path = os.path.join(d, "lessons")
        os.makedirs(path, exist_ok=True)
        return path


    def _new_lesson_dir(self):
        """
        Create a new lesson folder using timestamp.
        Example: 20251221_215700
        """
        base = self._profile_lessons_dir()
        if not base:
            return None
        lesson_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(base, lesson_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _profile_base_dir(self):
        return os.path.expanduser("~/.whisperx_diarize_gui")

    def _profile_dir(self):
        if not self.profile_name:
            return None
        return os.path.join(self._profile_base_dir(), "profiles", self.profile_name)

    def _profile_config_path(self):
        d = self._profile_dir()
        if not d:
            return None
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "config.json")

    def _load_profile_config(self):
        path = self._profile_config_path()
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _list_profile_lessons(self):
        base = self._profile_lessons_dir()
        if not base or not os.path.isdir(base):
            return []

        lessons = []
        for name in sorted(os.listdir(base), reverse=True):
            path = os.path.join(base, name)
            meta_path = os.path.join(path, "meta.json")
            if os.path.isdir(path) and os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    lessons.append((name, path, meta))
                except Exception:
                    continue
        return lessons

    def _start_mic_meter(self):
        self._mic_meter_running = True
        self._update_mic_meter()

    def _stop_mic_meter(self):
        self._mic_meter_running = False
        try:
            self.mic_level_bar.set(0.0)
            self.mic_level_db.configure(text="")
        except Exception:
            pass

    def _update_mic_meter(self):
        if not getattr(self, "_mic_meter_running", False):
            return

        # recorder may not exist or may not be recording
        try:
            rms, peak = self.recorder.get_level()  # <-- adapt name if different
        except Exception:
            rms, peak = 0.0, 0.0

        # Map RMS (0..1) to progress bar (0..1)
        level = max(0.0, min(1.0, rms * 4.0))  # scale so normal speech shows up
        self.mic_level_bar.set(level)

        # Optional dB estimate (avoid log(0))
        if rms > 1e-6:
            db = 20.0 * math.log10(rms)
            self.mic_level_db.configure(text=f"{db:0.0f} dB")
        else:
            self.mic_level_db.configure(text="")

        # Optional clip indicator (peak near 1.0)
        # if peak > 0.98: you could change label color or show "CLIP"
        # keep it simple initially

        self.master.after(60, self._update_mic_meter)  # ~16 fps

    def _save_profile_config(self, cfg: dict):
        path = self._profile_config_path()
        if not path:
            return
        # Keep permissions user-only if possible
        try:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            try:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Could not save profile config: {e}")

    def _cleanup_ollama(self):
        if hasattr(self, 'ollama_process') and self.ollama_process:
            self.ollama_process.terminate()
            self.ollama_process.wait()

    def _set_status(self, text):
        self.status_label.configure(text=f"Status: {text}")

    def _set_progress(self, value):
        # CTk progress bar is 0.0 to 1.0
        self.progress_bar.set(float(value) / 100.0)

    def _enable_export_buttons(self):
        self.analyze_btn.configure(state="normal")
        self.export_srt_btn.configure(state="normal")
        self.export_txt_btn.configure(state="normal")
        self.export_wav_btn.configure(state="normal")
        self.assign_btn.configure(state = "normal")
        self._update_analyze_ui_state()

    def _disable_export_buttons(self):
        self.analyze_btn.configure(state="disabled")
        self.export_srt_btn.configure(state="disabled")
        self.export_txt_btn.configure(state="disabled")
        self.export_wav_btn.configure(state="disabled")
        self._update_analyze_ui_state()


    def _read_json_file(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _write_json_file(self, path: str, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _reassociate_audio_for_lesson(self, lesson_dir: str, status_label=None):
        # Let user pick an audio file
        audio_path = filedialog.askopenfilename(
            title="Select the original audio file for this lesson",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.aiff *.aif"),
                ("All files", "*.*"),
            ],
        )
        if not audio_path:
            return

        meta_path = os.path.join(lesson_dir, "meta.json")
        meta = self._read_json_file(meta_path)

        meta["source_audio_path"] = audio_path

        # Optional: keep a timestamp
        meta["audio_reassociated_at"] = datetime.now().isoformat(timespec="seconds")

        self._write_json_file(meta_path, meta)

        # If this lesson is currently loaded, update pipeline immediately
        if getattr(self, "current_lesson_dir", None) == lesson_dir:
            try:
                if os.path.isfile(audio_path):
                    self.pipeline.last_audio_path = audio_path
            except Exception:
                pass

        # Update UI feedback
        if status_label is not None:
            try:
                status_label.configure(text=f"Audio: {os.path.basename(audio_path)}", text_color="lightgreen")
            except Exception:
                pass

        messagebox.showinfo("Updated", "Audio path saved. You can now Load this lesson and export speaker WAVs.")


    # --- PROFILE MANAGEMENT ---

    def _load_existing_profiles(self):
        base_dir = os.path.expanduser("~/.whisperx_diarize_gui")
        profiles_dir = os.path.join(base_dir, "profiles")
        if not os.path.isdir(profiles_dir):
            return []
        return sorted([n for n in os.listdir(profiles_dir) if os.path.isdir(os.path.join(profiles_dir, n))])

    def _prompt_profile_on_startup(self):
        existing = self._load_existing_profiles()
        
        win = ctk.CTkToplevel(self.master)
        win.title("Who is learning?")
        win.geometry("400x400")
        win.attributes("-topmost", True)
        
        ctk.CTkLabel(win, text="Select Profile", font=("Roboto", 20)).pack(pady=10)

        scroll = ctk.CTkScrollableFrame(win, height=200)
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        # Helper to set and close
        def select_and_close(name):
            self.profile_name = name
            self.profile_label.configure(text=f"Profile: {name}")
            self.profile_config = self._load_profile_config()
            
            # --- ADD THIS BLOCK ---
            # Reload Dashboard
            if self.dashboard:
                self.dashboard.destroy()
            
            self.dashboard = DashboardFrame(
                self.tab_dash, 
                profile_name=self.profile_name, 
                profile_dir=self._profile_dir(),
                pipeline=self.pipeline  
            )
            self.dashboard.pack(fill="both", expand=True)
            # ----------------------

            win.destroy()

        for name in existing:
            btn = ctk.CTkButton(scroll, text=name, fg_color="transparent", border_width=1, 
                                command=lambda n=name: select_and_close(n))
            btn.pack(fill="x", pady=2)

        ctk.CTkLabel(win, text="Or create new:").pack(pady=(10,0))
        entry = ctk.CTkEntry(win, placeholder_text="New Name")
        entry.pack(fill="x", padx=20, pady=5)

        def create_new():
            name = entry.get().strip()
            if name:
                select_and_close(name)

        ctk.CTkButton(win, text="Create & Start", command=create_new).pack(pady=10)
        win.wait_window() # Block until done

    def set_profile(self):
        dialog = ctk.CTkInputDialog(text="Enter new profile name:", title="New Profile")
        name = dialog.get_input()
        if name and name.strip():
            self.profile_name = name.strip()
            self.profile_label.configure(text=f"Profile: {self.profile_name}")
            self.profile_config = self._load_profile_config()


    # --- MAIN FUNCTIONS ---

    def select_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.m4a *.flac *.ogg"), ("All", "*.*")])
        if path:
            self.audio_path = path
            self.audio_label.configure(text=os.path.basename(path), text_color="white")

    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir = path
            self.output_label.configure(text=os.path.basename(path), text_color="white")

    def start_recording(self):
        
        if self.is_recording: return
        if not self.output_dir:
            self.select_output_dir()
            if not self.output_dir: return

        # Parse device
        dev_str = self.device_var.get()
        idx = int(dev_str.split(":")[0]) if ":" in dev_str else None
        
        self.recorder.start_recording(self.output_dir, device_index=idx)
        if self.recorder.is_recording:
            self.is_recording = True
            self._start_mic_meter()
            self.start_rec_btn.configure(state="disabled", fg_color="gray")
            self.stop_rec_btn.configure(state="normal", fg_color="#d32f2f")

    def stop_recording(self):
        if not self.is_recording: return
        f = self.recorder.stop_recording()
        self._stop_mic_meter()
        self.is_recording = False
        self.start_rec_btn.configure(state="normal", fg_color="#d32f2f")
        self.stop_rec_btn.configure(state="disabled", fg_color="gray")
        if f:
            self.audio_path = f
            self.audio_label.configure(text=os.path.basename(f), text_color="white")

    def run_diarization(self):
        if not self.audio_path or not self.output_dir:
            messagebox.showerror("Missing Info", "Please select audio and output folder.")
            return

        self.run_btn.configure(state="disabled")
        self._set_progress(0)
        self._set_status("Starting pipeline...")

        self.run_btn.configure(text="PROCESSING…", state="disabled")
        thread = threading.Thread(target=self._run_pipeline_thread)
        thread.daemon = True
        thread.start()

    def _run_pipeline_thread(self):
        try:
            lang = self.lang_var.get()
            lang_code = LANGUAGE_MAP.get(lang)
            exp = self.exp_spk_var.get() if hasattr(self, "exp_spk_var") else "Auto"
            num_speakers = None if exp == "Auto" else int(exp)
            
            self.pipeline.process_audio(
                audio_path=self.audio_path,
                output_dir=self.output_dir,
                model_size=self.model_var.get(),
                language=lang_code,
                num_speakers=num_speakers,
            )
            self.has_result = True
            self.master.after(0, self._enable_export_buttons)
            self.master.after(0, lambda: messagebox.showinfo("Done", "Processing Complete!"))
            self._set_status("Complete")

            try:
                self.current_lesson_dir = self._new_lesson_dir()
                if self.current_lesson_dir:
                    self.pipeline.save_lesson_artifacts(
                        self.current_lesson_dir,
                        profile_name=self.profile_name,
                        whisper_model_size=self.model_var.get(),
                        language=self.lang_var.get(),
                        contextual=self.context_var.get(),
                        extra_meta={"recorded_at": self.recorder.recorded_at_time}
                    )
                    self.master.after(0, self._enforce_speaker_assignment_after_save)
            except Exception as e:
                print(f"[WARN] Failed to save lesson artifacts: {e}")

        except Exception as e:
            self._set_status("Error")
            info = str(e)
            self.master.after(0, lambda: messagebox.showerror("Error", info))
        finally:
            self.master.after(0, lambda: self.run_btn.configure(state="normal", text="RUN PROCESSING"))
            

    def load_diarized_txt(self):
        path = filedialog.askopenfilename(filetypes=[("TXT", "*.txt")])
        if path:
            try:
                self.pipeline.load_segments_from_txt(path)
                self.output_dir = os.path.dirname(path)
                self.output_label.configure(text=os.path.basename(self.output_dir))
                self.has_result = True
                self._enable_export_buttons()
                self._set_status("TXT Loaded")
                self._update_analyze_ui_state()
                try:
                    self.current_lesson_dir = self._new_lesson_dir()
                    if self.current_lesson_dir:
                        self.pipeline.save_lesson_artifacts(
                            self.current_lesson_dir,
                            profile_name=self.profile_name,
                            whisper_model_size=None,
                            language=None,
                            contextual=None,
                        )
                        self.master.after(0, self._enforce_speaker_assignment_after_save)
                except Exception as e:
                    print(f"[WARN] Failed to save lesson artifacts from TXT: {e}")

            except Exception as e:
                messagebox.showerror("Error", str(e))

    # --- HISTORY VIEW ---

    def view_history(self):
        if not self.profile_name:
            messagebox.showerror("Error", "No profile selected.")
            return

        base_dir = os.path.expanduser("~/.whisperx_diarize_gui")
        lesson_dir = os.path.join(base_dir, "profiles", self.profile_name, "lessons")
        
        if not os.path.isdir(lesson_dir):
            messagebox.showinfo("History", "No history found for this profile.")
            return

        # Load lessons from artifact folders: lessons/<lesson_id>/meta.json
        lessons = []
        for entry in os.listdir(lesson_dir):
            path = os.path.join(lesson_dir, entry)
            meta_path = os.path.join(path, "meta.json")
            if os.path.isdir(path) and os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f) or {}
                    lessons.append({"lesson_id": entry, "lesson_dir": path, "meta": meta})
                except Exception:
                    pass

        # Sort newest first (prefer created_at; fallback to folder name)
        def _sort_key(x):
            meta = x.get("meta", {}) or {}
            return meta.get("created_at") or x.get("lesson_id") or ""

        lessons.sort(key=_sort_key, reverse=True)

        # UI
        win = ctk.CTkToplevel(self.master)
        win.title(f"History - {self.profile_name}")
        win.geometry("600x500")

        ctk.CTkLabel(win, text="Past Lessons", font=("Roboto", 20)).pack(pady=10)

        scroll = ctk.CTkScrollableFrame(win, fg_color="#2B2B2B")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        for item in lessons:
            meta = item.get("meta", {}) or {}
            lesson_id = item.get("lesson_id", "???")
            ldir = item.get("lesson_dir")

            ts = meta.get("recorded_at") or meta.get("processed_at") or meta.get("created_at") or lesson_id
            provider = meta.get("llm_provider", meta.get("provider", "—"))
            model = meta.get("llm_model", "—")
            dur = meta.get("duration_sec", 0.0)
            nspk = meta.get("num_speakers", "—")
            nseg = meta.get("num_segments", "—")

            card = ctk.CTkFrame(scroll, fg_color="#2B2B2B")
            card.pack(fill="x", pady=5)

            lbl = ctk.CTkLabel(
                card,
                text=f"Date: {ts}\nSpeakers: {nspk} | Segments: {nseg} | Duration: {dur/60:.1f} min\nLLM: {provider} / {model}",
                justify="left",
            )
            lbl.pack(side="left", padx=10, pady=8)

            btn_view = ctk.CTkButton(card,text="View", width=70, command=lambda d=ldir: self._open_lesson_detail(d))
            btn_view.pack(side="right", padx=(6, 10), pady=10)

            btn = ctk.CTkButton(
            card,
            text="Load",
            width=60,
            command=lambda d=ldir: self._load_lesson_into_app(d)
            )

            btn.pack(side="right", padx=10, pady=10)

    def _open_lesson_from_history(self, lesson_dir: str):
        print("OPEN LESSON DIR:", lesson_dir)
        print("FILES IN DIR:", os.listdir(lesson_dir))
        try:
            transcript_path = os.path.join(lesson_dir, "transcript.txt")
            analysis_path = os.path.join(lesson_dir, "analysis.txt")
            print("TRANSCRIPT PATH:", transcript_path, "exists?", os.path.isfile(transcript_path))
            print("ANALYSIS PATH:", analysis_path, "exists?", os.path.isfile(analysis_path))

            self.pipeline.load_lesson_artifacts(lesson_dir)
            self._set_status(f"Loaded lesson: {os.path.basename(lesson_dir)}")
            self._set_progress(0)

            # If you have these helpers, call them:
            try:
                self._update_analyze_ui_state()
            except Exception:
                pass

            # If your export buttons are enabled elsewhere, ensure they're enabled now:
            try:
                self.export_srt_btn.configure(state="normal")
                self.export_txt_btn.configure(state="normal")
                self.export_wav_btn.configure(state="normal")
            except Exception:
                pass


        except Exception as e:
            msg = str(e)
            self.master.after(0, lambda err=msg: messagebox.showerror("Error", err))

    def _load_lesson_into_app(self, lesson_dir: str):
        try:
            meta = self.pipeline.load_lesson_artifacts(lesson_dir)

            self.current_lesson_dir = lesson_dir
            self.output_dir = lesson_dir
            self.output_label.configure(text=os.path.basename(lesson_dir))

            self.has_result = True
            self._enable_export_buttons()
            self._update_analyze_ui_state()
            self._set_status(f"Loaded lesson {os.path.basename(lesson_dir)}")
            self._set_progress(0)

            # Make "Process" and other actions see an audio file as selected
            audio_path = getattr(self.pipeline, "last_audio_path", None)
            if audio_path:
                # use whichever attribute your app uses as the selected audio
                self.audio_path = audio_path  # <-- if you use self.audio_path
                self.selected_audio_path = audio_path  # <-- if you use this one
                # if you have a label showing selected file, update it:
                try:
                    self.audio_label.configure(text=os.path.basename(audio_path))
                except Exception:
                    pass

            # With option (2), audio might not exist anymore
            if not getattr(self.pipeline, "last_audio_path", None):
                messagebox.showwarning(
                    "Audio not found",
                    "This lesson was loaded, but the original audio file path is not available.\n\n"
                    "Speaker WAV export requires the original audio file. If you moved/deleted it, re-associate it."
                )
        except Exception as e:
            msg = str(e)
            self.master.after(0, lambda err=msg: messagebox.showerror("Load Failed", err))


    def _open_lesson_detail(self, lesson_dir: str):
        win = ctk.CTkToplevel(self.master)
        win.title("Lesson Detail")
        win.geometry("800x600")
        # --- Header row (buttons + audio status) ---
        header = ctk.CTkFrame(win)
        header.pack(fill="x", padx=10, pady=(10, 6))

        btn_relink = ctk.CTkButton(
            header,
            text="Re-associate audio…",
            width=160,
            command=lambda d=lesson_dir: self._reassociate_audio_for_lesson(d, status_label=audio_status),
        )
        # We reference audio_status below, so create it first (see next lines)
        meta_path = os.path.join(lesson_dir, "meta.json")
        meta = self._read_json_file(meta_path)
        ap = meta.get("source_audio_path")
        ap_ok = bool(ap and os.path.isfile(ap))
        audio_text = f"Audio: {os.path.basename(ap)}" if ap else "Audio: (not set)"
        audio_color = "lightgreen" if ap_ok else "gray"

        audio_status = ctk.CTkLabel(header, text=audio_text, text_color=audio_color)
        audio_status.pack(side="left", padx=(10, 8), pady=8)

        btn_relink = ctk.CTkButton(
            header,
            text="Re-associate audio…",
            width=160,
            command=lambda d=lesson_dir, lbl=audio_status: self._reassociate_audio_for_lesson(d, status_label=lbl),
        )
        btn_relink.pack(side="right", padx=10, pady=8)

        tabview = ctk.CTkTabview(win)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)

        t_tab = tabview.add("Transcript")
        a_tab = tabview.add("Analysis")

        # --- Transcript ---
        t_box = ctk.CTkTextbox(t_tab)
        t_box.pack(fill="both", expand=True)

        transcript_path = os.path.join(lesson_dir, "transcript.txt")
        if os.path.isfile(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                t_box.insert("1.0", f.read())
        else:
            t_box.insert("1.0", "[Transcript not found]")

        t_box.configure(state="disabled")

        # --- Analysis ---
        a_box = ctk.CTkTextbox(a_tab)
        a_box.pack(fill="both", expand=True)

        analysis_path = os.path.join(lesson_dir, "analysis.txt")
        if os.path.isfile(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as f:
                a_box.insert("1.0", f.read())
        else:
            a_box.insert("1.0", "No analysis available for this lesson.")

        a_box.configure(state="disabled")

    # --- ANALYSIS FLOW ---
    def analyze_transcript(self):
        if not self.has_result or not self.pipeline.last_result: return
        
        # Get speakers
        segs = self.pipeline.last_result["segments"]
        speakers = sorted(list(set(s.get("speaker", "") for s in segs if s.get("speaker"))))
        
        self._open_analysis_setup_window(speakers)

    def _open_analysis_setup_window(self, speakers):
        win = ctk.CTkToplevel(self.master)
        win.title("Setup Analysis")
        win.geometry("900x720")
        win.minsize(860, 680)

        # Container for all content (no window scrolling)
        container = ctk.CTkFrame(win, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=12, pady=5)
        
        # --- Cost estimate callback placeholder (defined later) ---
        def update_cost_estimate(_evt=None):
            return

        # --- Speakers + Preview side-by-side ---
        top_row = ctk.CTkFrame(container, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(5, 5))

        # Left column: Speakers
        left = ctk.CTkFrame(top_row)
        left.pack(side="left", fill="y", padx=(5, 5))

        ctk.CTkLabel(left, text="Speakers", font=("Roboto", 14, "bold")).pack(anchor="w", padx=12, pady=(5, 5))

        spk_frame = ctk.CTkScrollableFrame(left, width=220, height=220)
        spk_frame.pack(fill="y", padx=10, pady=(5, 5))

        # Right column: Transcript Preview
        right = ctk.CTkFrame(top_row)
        right.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(right, text="Transcript Preview (filtered)", font=("Roboto", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 6))

        preview_box = ctk.CTkTextbox(right, height=220)
        preview_box.pack(fill="both", expand=False, padx=10, pady=(0, 5))

        preview_meta = ctk.CTkLabel(right, text="", text_color="gray")
        preview_meta.pack(anchor="w", padx=12, pady=(0, 5))

        self.spk_vars = {}

        def update_preview():
            selected = [s for s, v in self.spk_vars.items() if v.get()]
            if not selected:
                text = "(Select at least one speaker to preview.)"
            else:
                try:
                    text = self.pipeline.get_transcript_text(
                        include_speaker=True,
                        speaker_filters=selected,
                        max_chars=6000,  # preview cap for responsiveness
                    )
                    if not text.strip():
                        text = "(No transcript text for the selected speaker(s).)"
                except Exception as e:
                    text = f"(Preview error: {e})"

            preview_box.configure(state="normal")
            preview_box.delete("1.0", "end")
            preview_box.insert("1.0", text)
            preview_box.configure(state="disabled")

            preview_meta.configure(text=f"Chars: {len(text)}")

            # If you have cost estimate logic, this is a good place to refresh it:
            try:
                update_cost_estimate()
            except Exception:
                pass

        # Now that update_preview exists, create checkboxes safely
        for spk in speakers:
            var = ctk.BooleanVar(value=(spk == "SPEAKER_00"))
            cb = ctk.CTkCheckBox(spk_frame, text=spk, variable=var, command=update_preview)
            cb.pack(anchor="w", pady=2, padx=5)
            self.spk_vars[spk] = var
        
        # Initialize preview
        update_preview()

        # 2. Model & Language
        # 2. Provider / Model / Language
        sett_frame = ctk.CTkFrame(container)
        sett_frame.pack(fill="x", padx=20, pady=5)

        # Provider selector
        self.analysis_provider_var = ctk.StringVar(
            value=self.profile_config.get("llm_provider", "ollama")
        )

        prov_row = ctk.CTkFrame(sett_frame, fg_color="transparent")
        prov_row.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkLabel(prov_row, text="LLM Provider:", font=("Roboto", 13, "bold")).pack(side="left", padx=(0, 5))

        ctk.CTkRadioButton(
            prov_row, text="Local (Ollama)",
            variable=self.analysis_provider_var, value="ollama",
            command=lambda: update_provider_ui()
        ).pack(side="left", padx=10)

        ctk.CTkRadioButton(
            prov_row, text="OpenAI",
            variable=self.analysis_provider_var, value="openai",
            command=lambda: update_provider_ui()
        ).pack(side="left", padx=10)

        # Model row
        model_row = ctk.CTkFrame(sett_frame, fg_color="transparent")
        model_row.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkLabel(model_row, text="Model:", width=60).pack(side="left")

        self.ollama_models = ["mistral", "mixtral", "gemma:2b", "llama3"]
        self.openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-5.1", "gpt-5.2"]
        
        # Pick model default from profile depending on provider
        provider0 = self.analysis_provider_var.get()
        if provider0 == "openai":
            default_model = self.profile_config.get("openai_model", "gpt-5.1")
        else:
            default_model = self.profile_config.get("ollama_model", "mistral")

        self.analysis_model_var = ctk.StringVar(value=default_model)

        self.model_menu = ctk.CTkOptionMenu(
            model_row,
            variable=self.analysis_model_var,
            values=self.openai_models if provider0 == "openai" else self.ollama_models,
            width=180,
            command=lambda _val=None: update_cost_estimate(),
        )

        self.model_menu.pack(side="left", padx=5)

        # OpenAI API key row (disabled unless OpenAI selected)
        openai_row = ctk.CTkFrame(sett_frame, fg_color="transparent")
        openai_row.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkLabel(openai_row, text="OpenAI API Key:", width=110, anchor="w").pack(side="left", padx=(0, 8))

        raw_key = self.profile_config.get("openai_api_key", "")
        self.openai_key_var = ctk.StringVar(value=deobfuscate_secret(raw_key))

        self.openai_key_entry = ctk.CTkEntry(
            openai_row,
            textvariable=self.openai_key_var,
            placeholder_text="sk-...",
            show="•",
            width=360
        )
        self.openai_key_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)

        self.save_key_var = ctk.BooleanVar(value=True)
        self.save_key_cb = ctk.CTkCheckBox(openai_row, text="Save to profile", variable=self.save_key_var)
        self.save_key_cb.pack(side="left")

        # Language row
        lang_row = ctk.CTkFrame(sett_frame, fg_color="transparent")
        lang_row.pack(fill="x", padx=10, pady=(5, 10))

        self.analysis_lang_var = ctk.StringVar(value=self.profile_config.get("analysis_lang", "EN"))
        ctk.CTkLabel(lang_row, text="Feedback:", width=60).pack(side="left")
        ctk.CTkRadioButton(lang_row, text="English", variable=self.analysis_lang_var, value="EN", command=update_cost_estimate).pack(side="left", padx=10)
        ctk.CTkRadioButton(lang_row, text="Spanish", variable=self.analysis_lang_var, value="ES", command=update_cost_estimate).pack(side="left", padx=10)

        def update_provider_ui():
            provider = self.analysis_provider_var.get()
            if provider == "openai":
                self.model_menu.configure(values=self.openai_models)
                if self.analysis_model_var.get() not in self.openai_models:
                    self.analysis_model_var.set(self.profile_config.get("openai_model", "gpt-4o"))
                self.openai_key_entry.configure(state="normal")
                self.save_key_cb.configure(state="normal")
            else:
                self.model_menu.configure(values=self.ollama_models)
                if self.analysis_model_var.get() not in self.ollama_models:
                    self.analysis_model_var.set(self.profile_config.get("ollama_model", "mistral"))
                self.openai_key_entry.configure(state="disabled")
                self.save_key_cb.configure(state="disabled")
            update_cost_estimate()


        # Apply initial enable/disable state
        update_provider_ui()
        # 3. Prompt
        ctk.CTkLabel(container, text="Custom Prompt:", font=("Roboto", 14, "bold")).pack(pady=(0,5))
        prompt_box = ctk.CTkTextbox(container, height=100)
        prompt_box.pack(fill="x", padx=20, expand=False)
        
        prompt_box.bind("<KeyRelease>", update_cost_estimate)
        prompt_box.bind("<FocusOut>", update_cost_estimate)

        default_prompt = (
            "You are an expert Spanish language teacher and pronunciation coach. \
            You will analyze ONLY the student’s speech from a lesson transcript. \
            Do not analyze or comment on the tutor’s speech. \
 \
            Your analysis must be objective, supportive, and pedagogically precise. \
            Base your feedback strictly on evidence in the transcript. \
\
            Produce your response in the following structured sections: \
 \
            1. Overall Assessment (2–3 sentences) \
            - Summarize the student’s current performance in terms of communicative effectiveness. \
            - State whether communication was generally smooth or effortful. \
 \
            2. Accuracy \
            - Identify recurring grammatical, morphological, or lexical errors. \
            - Focus on error patterns, not isolated slips. \
            - Provide 2–4 representative examples with corrections. \
            - Briefly explain why the correction is needed (no long grammar lessons). \
 \
            3. Fluency \
            - Evaluate flow, hesitation patterns, false starts, and sentence completion. \
            - Comment on whether pauses interfere with meaning or are natural at this level. \
\
            4. Pronunciation & Prosody \
            - Assess rhythm, syllable timing, stress, and intonation. \
            - Note any features that sound non-native (e.g., English stress patterns, vowel reduction). \
            - Also identify any aspects that already sound natural. \
\
            5. Positive Observations \
            - Highlight specific strengths demonstrated in this lesson \
            (e.g., verb tense control, use of connectors, conversational strategies). \
\
            6. CEFR Profile (Estimated) \
            Provide an estimated CEFR level for each dimension: \
            - Grammar accuracy \
            - Fluency \
            - Pronunciation & prosody \
            - Vocabulary range \
            Use labels such as: A2 / B1 / B2 / C1. \
            Briefly justify each estimate (1 sentence each). \
\
            7. Priority Practice Goals (Next 2–3 Weeks) \
            - List 3–5 concrete, actionable goals. \
            - Focus on the highest-impact improvements for the student. \
            - Phrase goals in practical terms (e.g., “practice linking clauses with…”, not “improve grammar”). \
\
            Guidelines: \
            - Do not rewrite the entire transcript. \
            - Do not invent errors not present in the text. \
            - Avoid vague advice (“practice more”, “be more fluent”). \
            - Assume the student is motivated and aiming for advanced proficiency.")
        
        prompt_box.insert("1.0", default_prompt)

        # --- Cost estimate (OpenAI only) ---
        est_frame = ctk.CTkFrame(container, fg_color="transparent")
        est_frame.pack(fill="x", padx=20, pady=(8, 0))

        self.cost_label = ctk.CTkLabel(
            est_frame,
            text="Estimated cost: (select OpenAI to estimate)",
            text_color="gray",
            justify="left",
        )
        self.cost_label.pack(anchor="w")

        # Try to use pipeline's default max chars if available, else fall back
        try:
            from .pipeline import DEFAULT_MAX_CHARS as _MAX_CHARS_DEFAULT
        except Exception:
            _MAX_CHARS_DEFAULT = 8000

        def _build_estimated_combined_prompt() -> str:
            # Speakers currently selected
            selected_spks = [s for s, v in self.spk_vars.items() if v.get()]
            speaker_filters = selected_spks if selected_spks else None

            # Prompt text + language instruction (matches what you'll send)
            base_prompt = prompt_box.get("1.0", "end").strip()
            lang = self.analysis_lang_var.get()
            if lang == "EN":
                base_prompt += "\n\nWrite response in ENGLISH."
            else:
                base_prompt += "\n\nEscribe la respuesta en ESPAÑOL."

            transcript = self.pipeline.get_transcript_text(
                include_speaker=True,
                speaker_filters=speaker_filters,
                max_chars=_MAX_CHARS_DEFAULT,
            )

            speakers_str = ", ".join(selected_spks) if selected_spks else "ALL"
            combined = (
                base_prompt
                + f"\n\n--- FILTERED TRANSCRIPT (speakers: {speakers_str}) ---\n"
                + transcript
            )
            return combined

        def update_cost_estimate(_evt=None):
            try:
                provider = self.analysis_provider_var.get()
            except Exception:
                provider = "ollama"

            if provider != "openai":
                self.cost_label.configure(
                    text="Estimated cost: (local analysis — no OpenAI cost)",
                    text_color="gray",
                )
                return

            model = self.analysis_model_var.get()
            combined_prompt = _build_estimated_combined_prompt()

            est = estimate_openai_cost(text=combined_prompt, model=model)
            if not est:
                self.cost_label.configure(
                    text=f"Estimated cost: (no pricing data for model: {model})",
                    text_color="gray",
                )
                return

            total = est["total_cost_usd"]
            inp = est["input_tokens"]
            out = est["output_tokens"]

            # Show a small range by varying output ratio (typical variance)
            est_low = estimate_openai_cost(text=combined_prompt, model=model, output_ratio=0.25)
            est_high = estimate_openai_cost(text=combined_prompt, model=model, output_ratio=0.50)
            if est_low and est_high:
                lo = est_low["total_cost_usd"]
                hi = est_high["total_cost_usd"]
                text = (
                    f"Estimated OpenAI cost: ~${lo:.4f} – ${hi:.4f} per run  "
                    f"(in ~{inp:,} tok, out ~{out:,} tok)"
                )
            else:
                text = (
                    f"Estimated OpenAI cost: ~${total:.4f} per run  "
                    f"(in ~{inp:,} tok, out ~{out:,} tok)"
                )

            # If it's getting expensive, use a warmer color
            color = "gray"
            if total >= 0.50:
                color = "orange"
            if total >= 2.00:
                color = "red"

            self.cost_label.configure(text=text, text_color=color)

        # Initialize estimate once the widgets exist
        update_cost_estimate()


        def on_run():
            selected_spks = [s for s, v in self.spk_vars.items() if v.get()]
            if not selected_spks:
                messagebox.showerror("Error", "Select at least one speaker.")
                return

            provider = self.analysis_provider_var.get()
            model = self.analysis_model_var.get()

            # Build prompt
            final_prompt = prompt_box.get("1.0", "end").strip()
            lang = self.analysis_lang_var.get()
            if lang == "EN":
                final_prompt += "\n\nWrite response in ENGLISH."
            else:
                final_prompt += "\n\nEscribe la respuesta en ESPAÑOL."

            # Load existing config and update preferences
            cfg = self._load_profile_config()
            cfg["llm_provider"] = provider
            cfg["analysis_lang"] = lang

            if provider == "openai":
                cfg["openai_model"] = model
            else:
                cfg["ollama_model"] = model

            # Handle provider-specific validation
            openai_key_to_use = None

            if provider == "openai":
                openai_key_to_use = (self.openai_key_var.get() or "").strip()
                if not openai_key_to_use:
                    messagebox.showerror("Error", "Please enter your OpenAI API key.")
                    return

                if self.save_key_var.get():
                    cfg["openai_api_key"] = obfuscate_secret(openai_key_to_use)

                self._save_profile_config(cfg)
                self.profile_config = cfg

                win.destroy()
                threading.Thread(
                    target=self._run_analysis_thread,
                    args=(final_prompt, selected_spks, provider, model, openai_key_to_use),
                    daemon=True
                ).start()
                return

            # provider == ollama (existing model availability check)
            api_url = os.environ.get("LLM_ANALYSIS_URL", "http://localhost:11434/api/generate")
            if not self.pipeline.check_ollama_model_availability(model, api_url):
                if messagebox.askyesno("Model Missing", f"Download {model}?"):
                    self._run_download_thread(model)
                return

            self._save_profile_config(cfg)
            self.profile_config = cfg

            win.destroy()
            threading.Thread(
                target=self._run_analysis_thread,
                args=(final_prompt, selected_spks, provider, model, None),
                daemon=True
            ).start()

        start_btn = ctk.CTkButton(container, text="Start Analysis", command=on_run, fg_color="#2e7d32", height=44)
        start_btn.pack(pady=(10, 20), fill="x", padx=10)

    def _speaker_map_path(self, lesson_dir: str) -> str:
        return os.path.join(lesson_dir, "speaker_map.json")

    def _has_speaker_assignment(self, lesson_dir: str) -> bool:
        return os.path.isfile(self._speaker_map_path(lesson_dir))

    def _run_download_thread(self, model_name):
        # Popup for download progress
        dl_win = ctk.CTkToplevel(self.master)
        dl_win.title(f"Downloading {model_name}...")
        dl_win.geometry("300x150")
        
        lbl = ctk.CTkLabel(dl_win, text="Downloading... please wait.")
        lbl.pack(pady=20)
        prog = ctk.CTkProgressBar(dl_win, mode="indeterminate")
        prog.pack(padx=20, fill="x")
        prog.start()

        def worker():
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.abspath(sys.executable))
            else:
                base_path = get_resource_base_path()
            
            ollama_bin = os.path.join(base_path, "deps", "ollama")
            if not os.path.exists(ollama_bin):
                ollama_bin = os.path.join(get_resource_base_path(), "ollama")

            env = os.environ.copy()
            env["OLLAMA_MODELS"] = os.path.expanduser("~/Library/Application Support/DiarizeApp/models")
            env["OLLAMA_HOST"] = "127.0.0.1:11435"

            try:
                subprocess.run([ollama_bin, "pull", model_name], env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.master.after(0, lambda: messagebox.showinfo("Success", "Model Installed!"))
            except Exception as e:
                msg = str(e)
                self.master.after(0, lambda: messagebox.showerror("Error", msg))
            finally:
                self.master.after(0, dl_win.destroy)

        threading.Thread(target=worker, daemon=True).start()

    def _enforce_speaker_assignment_after_save(self):
        """
        Call this after save_lesson_artifacts() for any workflow that creates/loads speaker-labeled segments.
        """
        if not getattr(self, "current_lesson_dir", None):
            return

        # Only enforce if there are speakers to assign
        seg_path = os.path.join(self.current_lesson_dir, "segments.json")
        if not os.path.isfile(seg_path):
            return

        try:
            with open(seg_path, "r", encoding="utf-8") as f:
                segs = json.load(f) or []
            speakers = sorted({s.get("speaker") for s in segs if s.get("speaker")})
        except Exception:
            speakers = []

        if not speakers:
            return

        diar_path = os.path.join(self.current_lesson_dir, "diarization.json")
        has_diar = os.path.isfile(diar_path)

        # Enforce if diarization exists OR multiple speakers exist
        if (has_diar or len(speakers) > 1) and not self._has_speaker_assignment(self.current_lesson_dir):
            if getattr(self, "_assign_window_open", False):
                return
            self._assign_window_open = True

            def _open():
                try:
                    self._set_status("Action required: Assign Student speaker(s).")
                    self._open_assign_speakers_window(speakers)
                finally:
                    self._assign_window_open = False

            self.master.after(0, _open)

    def _run_analysis_thread(self, prompt, speakers, provider, model, openai_api_key=None):
        self._set_status("Analyzing...")
        self._set_progress(20)
        try:
            res = self.pipeline.analyze_with_llm(
                user_prompt=prompt,
                provider=provider,
                model=model,
                api_key=openai_api_key if provider == "openai" else None,
                speakers=speakers,
            )

            # Ensure we have a lesson folder to attach analysis to
            if not getattr(self, "current_lesson_dir", None):
                self.current_lesson_dir = self._new_lesson_dir()
                if self.current_lesson_dir:
                    # If analysis happens without a saved transcript yet, save it now
                    self.pipeline.save_lesson_artifacts(
                        self.current_lesson_dir,
                        profile_name=self.profile_name,
                        whisper_model_size=self.model_var.get() if hasattr(self, "model_var") else None,
                        language=self.lang_var.get() if hasattr(self, "lang_var") else None,
                        contextual=self.context_var.get() if hasattr(self, "context_var") else None,
                    )

            # Write analysis.txt + patch meta.json
            if self.current_lesson_dir:
                analysis_path = os.path.join(self.current_lesson_dir, "analysis.txt")
                with open(analysis_path, "w", encoding="utf-8") as f:
                    f.write(res)

                meta_path = os.path.join(self.current_lesson_dir, "meta.json")
                meta = {}
                if os.path.isfile(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f) or {}

                meta["llm_provider"] = provider
                meta["llm_model"] = model
                meta["analysis_updated_at"] = datetime.now().isoformat(timespec="seconds")
                meta["has_analysis"] = True

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

            # Show result
            self.master.after(0, lambda text=res: self._show_analysis_window(text))
            self._set_status("Analysis Done")
            self._set_progress(100)

        except Exception as e:
            msg = str(e)
            self.master.after(0, lambda err=msg: messagebox.showerror("Analysis Failed", err))
            self._set_status("Error")


    def _show_analysis_window(self, text):
        win = ctk.CTkToplevel(self.master)
        win.title("Analysis Result")
        win.geometry("700x600")
        
        box = ctk.CTkTextbox(win)
        box.pack(fill="both", expand=True, padx=10, pady=10)
        box.insert("1.0", text)

    def _open_assign_speakers_window(self, speakers: list[str]):
        """
        Modal: user selects which diarized speakers correspond to the Student.
        Writes speaker_map.json into current_lesson_dir and patches meta.json.
        """
        if not getattr(self, "current_lesson_dir", None):
            messagebox.showerror("Error", "No current lesson directory set.")
            return

        lesson_dir = self.current_lesson_dir
        seg_path = os.path.join(lesson_dir, "segments.json")
        if not os.path.isfile(seg_path):
            messagebox.showerror("Error", f"Missing segments.json in:\n{lesson_dir}")
            return
      
        # Load segments once
        try:
            with open(seg_path, "r", encoding="utf-8") as f:
                segments = json.load(f) or []
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read segments.json:\n{e}")
            return

        # Helper to build preview text
        def build_preview(selected: set[str], max_chars: int = 6000) -> tuple[str, dict]:
            """
            Returns (preview_text, stats_dict).
            stats_dict includes char_count, num_segments, duration_sec (approx).
            """
            if not selected:
                return "[Select one or more student speakers to preview]\n", {
                    "char_count": 0,
                    "num_segments": 0,
                    "duration_sec": 0.0,
                }

            lines = []
            total_chars = 0
            nseg = 0
            min_t = None
            max_t = None

            for s in segments:
                spk = s.get("speaker")
                if spk not in selected:
                    continue

                txt = s.get("text") or ""
                if not txt:
                    continue

                st = s.get("start")
                en = s.get("end")

                # track duration bounds
                if isinstance(st, (int, float)):
                    min_t = st if min_t is None else min(min_t, st)
                if isinstance(en, (int, float)):
                    max_t = en if max_t is None else max(max_t, en)

                # timestamp format mm:ss
                def fmt(t):
                    m = int(t // 60)
                    sec = int(t % 60)
                    return f"{m:02d}:{sec:02d}"

                ts = ""
                if isinstance(st, (int, float)) and isinstance(en, (int, float)):
                    ts = f"[{fmt(st)}–{fmt(en)}] "

                line = f"{ts}{spk}: {txt}".strip()

                if total_chars + len(line) + 1 > max_chars:
                    lines.append("… [preview truncated]")
                    break

                lines.append(line)
                total_chars += len(line) + 1
                nseg += 1

            dur = 0.0
            if min_t is not None and max_t is not None:
                dur = max(0.0, float(max_t) - float(min_t))

            return ("\n".join(lines) + ("\n" if lines else "")), {
                "char_count": total_chars,
                "num_segments": nseg,
                "duration_sec": dur,
            }


        # Pre-select from existing mapping if present
        speaker_map_path = os.path.join(lesson_dir, "speaker_map.json")
        pre_selected = set()
        if os.path.isfile(speaker_map_path):
            try:
                with open(speaker_map_path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
                pre_selected = set(existing.get("student_speakers") or [])
            except Exception:
                pre_selected = set()

        # --- Window ---
        win = ctk.CTkToplevel(self.master)
        win.title("Assign Student Speakers")
        win.geometry("900x560")
        win.minsize(860, 520)
        win.transient(self.master)
        win.grab_set()  # modal

        # Title + help
        header = ctk.CTkFrame(win, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=(12, 6))

        ctk.CTkLabel(header, text="Assign Student Speaker(s)", font=("Roboto", 18, "bold")).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Select which diarized speaker labels correspond to the STUDENT. This is required for exports and analytics.",
            text_color="gray",
        ).pack(anchor="w", pady=(2, 0))

        # --- Top row: Speakers (left) + Preview (right) ---
        top = ctk.CTkFrame(win)
        top.pack(fill="both", expand=True, padx=14, pady=(6, 10))

        left = ctk.CTkFrame(top)
        left.pack(side="left", fill="y", padx=(10, 8), pady=10)

        ctk.CTkLabel(left, text="Speakers", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10, pady=(8, 6))

        spk_scroll = ctk.CTkScrollableFrame(left, width=240, height=320)
        spk_scroll.pack(fill="y", padx=10, pady=(0, 10))

        right = ctk.CTkFrame(top)
        right.pack(side="left", fill="both", expand=True, padx=(8, 10), pady=10)

        ctk.CTkLabel(right, text="Transcript Preview (Student only)", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(8, 6)
        )

        preview_box = ctk.CTkTextbox(right, height=320)
        preview_box.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        preview_box.insert("1.0", "[Select one or more student speakers to preview]\n")
        preview_box.configure(state="disabled")

        preview_meta = ctk.CTkLabel(right, text="Chars: 0", text_color="gray")
        preview_meta.pack(anchor="w", padx=10, pady=(0, 6))

        # State vars
        spk_vars: dict[str, ctk.BooleanVar] = {}

        def update_preview(_evt=None):
            selected = {spk for spk, v in spk_vars.items() if v.get()}
            text, stats = build_preview(selected)

            preview_box.configure(state="normal")
            preview_box.delete("1.0", "end")
            preview_box.insert("1.0", text)
            preview_box.configure(state="disabled")

            preview_meta.configure(
                text=f"Chars: {stats['char_count']}   Segments: {stats['num_segments']}   ~Duration: {stats['duration_sec']/60:.1f} min"
            )

            btn_save.configure(state="normal" if selected else "disabled")


        # Build speaker checkboxes
        for spk in speakers:
            var = ctk.BooleanVar(value=(spk in pre_selected))
            spk_vars[spk] = var
            cb = ctk.CTkCheckBox(spk_scroll, text=spk, variable=var, command=update_preview)
            cb.pack(anchor="w", padx=10, pady=6)

        # Initialize preview
        self.master.after(0, update_preview)

        # --- Bottom row: buttons ---
        bottom = ctk.CTkFrame(win)
        bottom.pack(fill="x", padx=14, pady=(0, 14))

        # Status / warning line
        status_lbl = ctk.CTkLabel(bottom, text="", text_color="gray")
        status_lbl.pack(side="left", padx=10, pady=10)

        def do_save():
            selected = [spk for spk, v in spk_vars.items() if v.get()]
            if not selected:
                messagebox.showwarning("Required", "Please select at least one Student speaker.")
                return

            # Write speaker_map.json
            payload = {
                "version": 1,
                "assigned_at": datetime.now().isoformat(timespec="seconds"),
                "student_speakers": selected,
                "num_student_speakers": len(selected),
                "map": {spk: ("student" if spk in selected else "other") for spk in speakers},
            }
            try:
                with open(speaker_map_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to write speaker_map.json:\n{e}")
                return

            # Patch meta.json
            meta_path = os.path.join(lesson_dir, "meta.json")
            meta = {}
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f) or {}
                except Exception:
                    meta = {}

            meta["has_speaker_map"] = True
            meta["student_speakers"] = selected
            meta["speaker_map_file"] = "speaker_map.json"
            meta["speaker_map_updated_at"] = datetime.now().isoformat(timespec="seconds")

            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update meta.json:\n{e}")
                return

            win.grab_release()
            win.destroy()
            self.master.after(0, self._update_analyze_ui_state)

        def do_cancel():
            # If you truly want to *insist*, you can remove Cancel entirely.
            # For now: allow cancel but keep gating active (buttons disabled until assigned).
            self.master.after(0, self._update_analyze_ui_state)
            win.grab_release()
            win.destroy()

        btn_cancel = ctk.CTkButton(bottom, text="Close", width=120, fg_color="transparent", border_width=2, command=do_cancel)
        btn_cancel.pack(side="right", padx=(6, 10), pady=10)

        btn_save = ctk.CTkButton(bottom, text="Save Assignment", width=180, command=do_save)
        btn_save.pack(side="right", padx=(10, 6), pady=10)

        # Disable save until selection
        btn_save.configure(state="normal" if pre_selected else "disabled")

    # --- EXPORTS ---

    def export_srt(self):
        path = filedialog.asksaveasfilename(defaultextension=".srt")
        if path:
            self.pipeline.export_srt(path)
            messagebox.showinfo("Export", "Saved SRT")

    def export_txt(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            self.pipeline.export_txt(path)
            messagebox.showinfo("Export", "Saved TXT")

    def export_speaker_audio(self):
        if self.current_lesson_dir and not self._has_speaker_assignment(self.current_lesson_dir):
            messagebox.showwarning("Action required", "Please assign Student speaker(s) first.")
            self._enforce_speaker_assignment_after_save()
            return

        path = filedialog.askdirectory()
        if path:
            self.pipeline.export_speaker_audios(path)
            messagebox.showinfo("Export", "Saved Speaker Audios")


def main():
    root = ctk.CTk()
    app = DiarizationApp(root)
    
    def on_closing():
        # 1. Stop Ollama
        app._cleanup_ollama()
        
        # 2. Force Matplotlib to close all charts
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass

        # 3. Destroy the GUI
        root.destroy()
        
        # 4. Force kill the process (ensures no background threads hang)
        import sys
        sys.exit(0)

    # Bind the window's "X" button to our cleanup function
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()