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

class DiarizationApp:
    def __init__(self, master):
        self.master = master
        self.profile_config = {}

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
        self._build_ui()

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

    def _build_ui(self):
        # 1. PROFILE HEADER
        self.profile_frame = ctk.CTkFrame(self.master, corner_radius=10)
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
        self.input_card = ctk.CTkFrame(self.master)
        self.input_card.pack(padx=15, pady=5, fill="x")
        
        ctk.CTkLabel(self.input_card, text="Input Source", font=("Roboto", 14, "bold")).pack(anchor="w", padx=15, pady=(10,5))

        # A. File Select
        self.file_row = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.file_row.pack(fill="x", padx=10, pady=5)
        
        self.audio_btn = ctk.CTkButton(self.file_row, text="Select Audio", image=self.icon_folder, command=self.select_audio, width=120)
        self.audio_btn.pack(side="left", padx=5)
        
        self.audio_label = ctk.CTkLabel(self.file_row, text="(No file selected)", text_color="gray")
        self.audio_label.pack(side="left", padx=5)

        ctk.CTkLabel(self.input_card, text="- OR -", text_color="gray", font=("Arial", 10)).pack()

        # B. Recording
        self.rec_row = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.rec_row.pack(fill="x", padx=10, pady=5)

        # Device list
        self.input_devices = self.recorder.list_input_devices()
        dev_names = ["(default)"] + [f"{d['index']}: {d['name']}" for d in self.input_devices]
        
        self.device_var = ctk.StringVar(value="(default)")
        self.device_menu = ctk.CTkOptionMenu(self.rec_row, variable=self.device_var, values=dev_names, width=220)
        self.device_menu.pack(side="left", padx=5)

        self.start_rec_btn = ctk.CTkButton(
            self.rec_row, text="REC", width=60, 
            fg_color="#d32f2f", hover_color="#b71c1c", 
            image=self.icon_mic, command=self.start_recording
        )
        self.start_rec_btn.pack(side="left", padx=5)
        
        self.stop_rec_btn = ctk.CTkButton(self.rec_row, text="STOP", width=60, state="disabled", command=self.stop_recording)
        self.stop_rec_btn.pack(side="left", padx=5)

        # 3. SETTINGS CARD
        self.settings_card = ctk.CTkFrame(self.master)
        self.settings_card.pack(padx=15, pady=10, fill="x")
        
        ctk.CTkLabel(self.settings_card, text="Processing Settings", font=("Roboto", 14, "bold")).pack(anchor="w", padx=15, pady=(10,5))
        
        grid = ctk.CTkFrame(self.settings_card, fg_color="transparent")
        grid.pack(fill="x", padx=10, pady=5)

        # Output folder
        self.out_btn = ctk.CTkButton(grid, text="Output Folder", width=120, command=self.select_output_dir)
        self.out_btn.grid(row=0, column=0, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(grid, text="(None)", text_color="gray")
        self.output_label.grid(row=0, column=1, padx=5, sticky="w")

        # Model Size
        ctk.CTkLabel(grid, text="Model Size:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.model_var = ctk.StringVar(value="small")
        self.model_menu = ctk.CTkOptionMenu(grid, variable=self.model_var, values=["tiny", "base", "small", "medium", "large-v2"], width=100)
        self.model_menu.grid(row=1, column=1, padx=5, sticky="w")

        # Language
        ctk.CTkLabel(grid, text="Language:").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.lang_var = ctk.StringVar(value="Auto-Detect")
        self.lang_combo = ctk.CTkOptionMenu(grid, variable=self.lang_var, values=list(LANGUAGE_MAP.keys()), width=120)
        self.lang_combo.grid(row=1, column=3, padx=5, sticky="w")
        
        # Checkbox
        self.condition_checkbox = ctk.CTkCheckBox(self.settings_card, text="Condition on previous text (Context)", onvalue=True, offvalue=False)
        self.condition_checkbox.pack(padx=20, pady=(5, 15), anchor="w")

        # 4. ACTION
        self.run_btn = ctk.CTkButton(self.master, text="RUN PROCESSING", height=50, font=("Roboto", 18, "bold"), command=self.run_diarization)
        self.run_btn.pack(padx=15, pady=5, fill="x")

        # Status & Progress
        self.progress_bar = ctk.CTkProgressBar(self.master)
        self.progress_bar.pack(padx=15, pady=(5,5), fill="x")
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self.master, text="Status: Idle", text_color="gray")
        self.status_label.pack(padx=15, pady=(0,10), anchor="w")

        # 5. POST-PROCESSING CARD
        self.post_card = ctk.CTkFrame(self.master)
        self.post_card.pack(padx=15, pady=5, fill="x")

        self.analyze_btn = ctk.CTkButton(
            self.post_card, 
            text="Analyze with AI Assistant", 
            state="disabled", 
            fg_color="#2e7d32", 
            hover_color="#1b5e20",
            height=40,
            command=self.analyze_transcript
        )
        self.analyze_btn.pack(padx=10, pady=10, fill="x")

        exp_row = ctk.CTkFrame(self.post_card, fg_color="transparent")
        exp_row.pack(fill="x", padx=10, pady=(0,10))
        
        self.export_srt_btn = ctk.CTkButton(exp_row, text="Export SRT", state="disabled", width=80, command=self.export_srt)
        self.export_srt_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        self.export_txt_btn = ctk.CTkButton(exp_row, text="Export TXT", state="disabled", width=80, command=self.export_txt)
        self.export_txt_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        self.export_wav_btn = ctk.CTkButton(exp_row, text="Export Speakers", state="disabled", width=80, command=self.export_speaker_audio)
        self.export_wav_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        # Load TXT Button (at bottom)
        self.load_txt_btn = ctk.CTkButton(self.master, text="Load Existing TXT", fg_color="transparent", border_width=1, text_color=("gray10", "gray90"), command=self.load_diarized_txt)
        self.load_txt_btn.pack(pady=10)

    # --- LOGIC METHODS ---
    # --- PROFILE CONFIG (OpenAI key, provider prefs) ---

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

    def _disable_export_buttons(self):
        self.analyze_btn.configure(state="disabled")
        self.export_srt_btn.configure(state="disabled")
        self.export_txt_btn.configure(state="disabled")
        self.export_wav_btn.configure(state="disabled")

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
             # Load profile config (OpenAI key, prefs)
            self.profile_config = self._load_profile_config()

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
            self.start_rec_btn.configure(state="disabled", fg_color="gray")
            self.stop_rec_btn.configure(state="normal", fg_color="#d32f2f")

    def stop_recording(self):
        if not self.is_recording: return
        f = self.recorder.stop_recording()
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
        
        thread = threading.Thread(target=self._run_pipeline_thread)
        thread.daemon = True
        thread.start()

    def _run_pipeline_thread(self):
        try:
            lang = self.lang_var.get()
            lang_code = LANGUAGE_MAP.get(lang)
            
            self.pipeline.process_audio(
                audio_path=self.audio_path,
                output_dir=self.output_dir,
                model_size=self.model_var.get(),
                language=lang_code
            )
            self.has_result = True
            self.master.after(0, self._enable_export_buttons)
            self.master.after(0, lambda: messagebox.showinfo("Done", "Processing Complete!"))
            self._set_status("Complete")
        except Exception as e:
            self._set_status("Error")
            self.master.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.master.after(0, lambda: self.run_btn.configure(state="normal"))

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

        # Load JSONs
        lessons = []
        for fname in os.listdir(lesson_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(lesson_dir, fname), "r") as f:
                        data = json.load(f)
                        lessons.append(data)
                except: pass
        
        # Sort by timestamp
        lessons.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # UI
        win = ctk.CTkToplevel(self.master)
        win.title(f"History - {self.profile_name}")
        win.geometry("600x500")

        ctk.CTkLabel(win, text="Past Lessons", font=("Roboto", 20)).pack(pady=10)

        scroll = ctk.CTkScrollableFrame(win)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        for l in lessons:
            ts = l.get("timestamp", "???")
            provider = l.get("llm_provider", "ollama")
            model = l.get("llm_model", "unknown")
            # Create a card for each lesson
            card = ctk.CTkFrame(scroll, fg_color="gray25")
            card.pack(fill="x", pady=5)
            
            lbl = ctk.CTkLabel(card, text=f"Date: {ts}\nProvider: {provider}\nModel: {model}", justify="left")
            lbl.pack(side="left", padx=10, pady=5)
            
            btn = ctk.CTkButton(card, text="Open", width=60, command=lambda d=l: self._open_lesson_detail(d))
            btn.pack(side="right", padx=10)

    def _open_lesson_detail(self, data):
        win = ctk.CTkToplevel(self.master)
        win.title("Lesson Detail")
        win.geometry("800x600")

        # Tabs for Transcript vs Analysis
        tabview = ctk.CTkTabview(win)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        t_tab = tabview.add("Transcript")
        a_tab = tabview.add("Analysis")
        
        # Transcript Tab
        t_box = ctk.CTkTextbox(t_tab)
        t_box.pack(fill="both", expand=True)
        t_box.insert("1.0", data.get("transcript_text", ""))
        t_box.configure(state="disabled")
        
        # Analysis Tab
        a_box = ctk.CTkTextbox(a_tab)
        a_box.pack(fill="both", expand=True)
        a_box.insert("1.0", data.get("llm_response", ""))
        a_box.configure(state="disabled")

    # --- ANALYSIS FLOW ---
    def update_preview():
        selected = [s for s, v in self.spk_vars.items() if v.get()]
        if not selected:
            text = "(Select at least one speaker to preview.)"
        else:
            # You can pick a smaller max for preview to keep UI snappy
            try:
                text = self.pipeline.get_transcript_text(
                    include_speaker=True,
                    speaker_filters=selected,
                    max_chars=6000,
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
        update_cost_estimate()

    def analyze_transcript(self):
        if not self.has_result or not self.pipeline.last_result: return
        
        # Get speakers
        segs = self.pipeline.last_result["segments"]
        speakers = sorted(list(set(s.get("speaker", "") for s in segs if s.get("speaker"))))
        
        self._open_analysis_setup_window(speakers)

    def _open_analysis_setup_window(self, speakers):
        win.title("Setup Analysis")
        win.geometry("780x900")
        win.minsize(760, 820)

        # NEW: Scroll container for all content
        container = ctk.CTkScrollableFrame(container)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        
        # --- Cost estimate callback placeholder (defined later) ---
        def update_cost_estimate(_evt=None):
            return

        # 1. Speaker Selection
        ctk.CTkLabel(container, text="Select STUDENT Speakers:", font=("Roboto", 14, "bold")).pack(pady=(10,5))
        
        spk_frame = ctk.CTkScrollableFrame(container, height=150)
        spk_frame.pack(fill="x", padx=20)
        
        preview_box = ctk.CTkTextbox(container, height=220)
        preview_box.pack(fill="both", expand=False, padx=5, pady=(0, 10))

        preview_meta = ctk.CTkLabel(container, text="", text_color="gray")
        preview_meta.pack(anchor="w", padx=6, pady=(0, 10))

        self.spk_vars = {}
        for spk in speakers:
            var = ctk.BooleanVar(value=(spk == "SPEAKER_00"))
            cb = ctk.CTkCheckBox(spk_frame, text=spk, variable=var, command=update_preview)
            cb.pack(anchor="w", pady=2)
            self.spk_vars[spk] = var

        # --- Transcript Preview ---
        ctk.CTkLabel(container, text="Transcript Preview (filtered):", font=("Roboto", 14, "bold")).pack(pady=(10, 5))
        update_preview()

        # 2. Model & Language
        # 2. Provider / Model / Language
        sett_frame = ctk.CTkFrame(container)
        sett_frame.pack(fill="x", padx=20, pady=10)

        # Provider selector
        self.analysis_provider_var = ctk.StringVar(
            value=self.profile_config.get("llm_provider", "ollama")
        )

        prov_row = ctk.CTkFrame(sett_frame, fg_color="transparent")
        prov_row.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(prov_row, text="LLM Provider:", font=("Roboto", 13, "bold")).pack(side="left", padx=(0, 10))

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
        openai_row.pack(fill="x", padx=10, pady=(5, 5))

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
        ctk.CTkLabel(container, text="Custom Prompt:", font=("Roboto", 14, "bold")).pack(pady=(10,5))
        prompt_box = ctk.CTkTextbox(container, height=200)
        prompt_box.pack(fill="x", padx=20)
        
        prompt_box.bind("<KeyRelease>", update_cost_estimate)
        prompt_box.bind("<FocusOut>", update_cost_estimate)


        default_prompt = (
            "You are an expert Spanish teacher. Identify the student's mistakes "
            "and provide corrections and practice goals."
        )
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
                self.master.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.master.after(0, dl_win.destroy)

        threading.Thread(target=worker, daemon=True).start()

    def _run_analysis_thread(self, prompt, speakers, provider, model, openai_api_key=None):
        self._set_status("Analyzing...")
        self._set_progress(20) # fake progress
        try:
            res = self.pipeline.analyze_with_llm(
                user_prompt=prompt,
                provider=provider,
                model=model,
                api_key=openai_api_key if provider == "openai" else None,
                speakers=speakers,
            )


            self._save_lesson_record(prompt, res, model, provider=provider)
            self.master.after(0, lambda: self._show_analysis_window(res))
            self._set_status("Analysis Done")
            self._set_progress(100)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Analysis Failed", str(e)))
            self._set_status("Error")

    def _show_analysis_window(self, text):
        win = ctk.CTkToplevel(self.master)
        win.title("Analysis Result")
        win.geometry("700x600")
        
        box = ctk.CTkTextbox(win)
        box.pack(fill="both", expand=True, padx=10, pady=10)
        box.insert("1.0", text)

    def _save_lesson_record(self, prompt, response, model, provider=None):
        if not self.profile_name: return
        base_dir = os.path.expanduser("~/.whisperx_diarize_gui")
        d = os.path.join(base_dir, "profiles", self.profile_name, "lessons")
        os.makedirs(d, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = {
            "timestamp": ts,
            "llm_model": model,
            "llm_prompt": prompt,
            "llm_response": response,
            "llm_provider": provider or "ollama",
            "transcript_text": self.pipeline.get_transcript_text(include_speaker=True)
        }
        with open(os.path.join(d, f"lesson_{ts}.json"), "w") as f:
            json.dump(record, f, indent=2)

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
        path = filedialog.askdirectory()
        if path:
            self.pipeline.export_speaker_audios(path)
            messagebox.showinfo("Export", "Saved Speaker Audios")

def main():
    root = ctk.CTk()
    app = DiarizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()