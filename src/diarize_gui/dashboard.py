import os
import json
import re
import threading
from datetime import datetime, timedelta
from collections import defaultdict
import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from .theme import AppTheme
from .pipeline import DEFAULT_OLLAMA_ANALYSIS_MODEL
from .lesson_selection import select_all_incomplete_ai_lesson_dirs, select_pending_ai_lesson_dirs
from .metrics.context_adjusted import build_context_metrics, compute_automaticity_gap
from .utils import deobfuscate_secret

# Use a safe backend for macOS/Windows
matplotlib.use("TkAgg")


class DashboardToolTip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self.after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")

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
            wraplength=320,
        )
        label.pack()

    def _hide(self, _=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class DashboardFrame(ctk.CTkFrame):
    def __init__(self, master, profile_name, profile_dir, pipeline=None, **kwargs):
        super().__init__(master, **kwargs)
        self.profile_name = str(profile_name) if profile_name else "Student"
        self.profile_dir = profile_dir
        self.pipeline = pipeline 
        self.current_lesson_dir = None
        
        # Colors
        self.color_primary = AppTheme.BTN_PRIMARY
        self.color_student = AppTheme.BTN_SUCCESS
        self.color_tutor = "#C98A1A"
        self.color_bad    = "#D76A5D"
        self.color_ai     = "#9A4BCF"
        self.bg_figure = "#2b2b2b" if ctk.get_appearance_mode() == "Dark" else "#ffffff"
        self.text_color = AppTheme.TEXT_PRIMARY
        self.card_bg = AppTheme.BG_CARD
        self.card_border = AppTheme.BORDER_DIVIDER
        self.card_subtext = AppTheme.TEXT_MUTED

        self.filler_pattern = re.compile(r"\b(um|uh|eh|mm|hm|este|em)\b", re.IGNORECASE)

        # --- UI LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # Header
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 4))
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text=f"{self.profile_name} Dashboard",
            font=("Roboto", 22, "bold"),
            text_color=self.color_primary,
            anchor="w",
        )
        self.title_label.grid(row=0, column=0, sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="Conversation trends, fluency, and AI feedback in one place.",
            font=("Roboto", 12),
            text_color=self.card_subtext,
            anchor="w",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.profile_pill = ctk.CTkLabel(
            self.header_frame,
            text="LIVE PROFILE",
            font=("Roboto", 10, "bold"),
            text_color=AppTheme.BTN_TEXT_ON_BLUE,
            fg_color=self.color_primary,
            corner_radius=999,
            padx=10,
            pady=4,
        )
        self.profile_pill.grid(row=0, column=1, rowspan=2, sticky="e")

        # 1. Standard KPI Row
        self.row1 = ctk.CTkFrame(self, fg_color="transparent")
        self.row1.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 6))
        
        self.card_total_time = self._create_kpi_card(self.row1, "Total Hours", "0.0")
        self.card_student_pct = self._create_kpi_card(self.row1, "You Spoke", "0%")
        self.card_wpm = self._create_kpi_card(self.row1, "Your WPM", "0")
        self.card_words = self._create_kpi_card(self.row1, "Total Words", "0")

        self.card_total_time.pack(side="left", expand=True, fill="x", padx=5)
        self.card_student_pct.pack(side="left", expand=True, fill="x", padx=5)
        self.card_wpm.pack(side="left", expand=True, fill="x", padx=5)
        self.card_words.pack(side="left", expand=True, fill="x", padx=5)

        # 2. Advanced / AI KPI Row
        self.row2 = ctk.CTkFrame(self, fg_color="transparent")
        self.row2.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        
        self.card_grammar = self._create_kpi_card(self.row2, "Avg Grammar", "--", color=self.color_ai)
        self.card_auto_gap = self._create_kpi_card(self.row2, "Automaticity Gap", "--", color=self.color_ai)
        self.card_latency = self._create_kpi_card(self.row2, "Avg Latency", "0.0s")
        self.card_max_turn = self._create_kpi_card(self.row2, "Longest Turn", "0s")

        self.card_grammar.pack(side="left", expand=True, fill="x", padx=5)
        self.card_auto_gap.pack(side="left", expand=True, fill="x", padx=5)
        self.card_latency.pack(side="left", expand=True, fill="x", padx=5)
        self.card_max_turn.pack(side="left", expand=True, fill="x", padx=5)

        # Golden words panel
        self.golden_panel = ctk.CTkFrame(
            self,
            fg_color=self.card_bg,
            corner_radius=14,
            border_width=1,
            border_color=self.card_border,
        )
        self.golden_panel.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        self.golden_panel.grid_columnconfigure(0, weight=1)

        self.golden_header = ctk.CTkLabel(
            self.golden_panel,
            text="Recent Golden Words",
            font=("Roboto", 12, "bold"),
            text_color=self.color_ai,
            anchor="w",
        )
        self.golden_header.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))

        self.golden_hint = ctk.CTkLabel(
            self.golden_panel,
            text="Latest vocabulary targets pulled from AI analysis.",
            font=("Roboto", 11),
            text_color=self.card_subtext,
            anchor="w",
        )
        self.golden_hint.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 8))

        self.golden_words_container = ctk.CTkFrame(self.golden_panel, fg_color="transparent")
        self.golden_words_container.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.golden_words_container.grid_columnconfigure(0, weight=1)
        self.golden_words_container.grid_columnconfigure(1, weight=1)
        self.golden_words_container.grid_columnconfigure(2, weight=1)

        self.golden_word_labels = []
        for idx in range(3):
            lbl = ctk.CTkLabel(
                self.golden_words_container,
                text="--",
                fg_color=AppTheme.BG_ELEVATED,
                text_color=self.text_color,
                corner_radius=10,
                padx=12,
                pady=8,
                justify="center",
                anchor="center",
                wraplength=220,
            )
            lbl.grid(row=0, column=idx, sticky="ew", padx=4)
            self.golden_word_labels.append(lbl)

        # 3. Charts Area (Now Tabbed!)
        self.chart_tabs = ctk.CTkTabview(self)
        self.chart_tabs.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=12, pady=6)
        
        self.tab_activity = self.chart_tabs.add("Activity")
        self.tab_fluency = self.chart_tabs.add("Fluency")
        self.tab_grammar = self.chart_tabs.add("Grammar AI")
        self.tab_context = self.chart_tabs.add("Context")
        
        # 4. Controls Row
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=5, column=0, columnspan=2, pady=(8, 12))

        self.refresh_btn = ctk.CTkButton(
            self.controls_frame,
            text="Refresh Dashboard",
            command=self.refresh_data,
            fg_color=self.color_primary,
            hover_color=AppTheme.BTN_PRIMARY_HOVER,
        )
        self.refresh_btn.pack(side="left", padx=10)

        self.ai_btn = ctk.CTkButton(
            self.controls_frame,
            text="Compute Recent AI Metrics",
            fg_color=self.color_ai,
            hover_color="#7B1FA2",
            command=self.run_ai_analysis,
        )
        self.ai_btn.pack(side="left", padx=10)

        self.ai_backfill_btn = ctk.CTkButton(
            self.controls_frame,
            text="Backfill All AI Metrics",
            fg_color="#7B1FA2",
            hover_color="#5E167A",
            command=self.run_ai_backfill,
        )
        self.ai_backfill_btn.pack(side="left", padx=10)
        
        self.status_lbl = ctk.CTkLabel(self.controls_frame, text="", text_color=self.card_subtext)
        self.status_lbl.pack(side="left", padx=10)

        self._install_tooltips()

        # Initial Load
        self.refresh_data()

    def set_current_lesson_dir(self, lesson_dir):
        self.current_lesson_dir = lesson_dir

    def refresh_from_current_lesson(self, lesson_dir=None):
        """
        Convenience hook for the app to tell the dashboard which lesson is active,
        then refresh the aggregate view.
        """
        if lesson_dir is not None:
            self.current_lesson_dir = lesson_dir
        self.refresh_data()

    def _create_kpi_card(self, parent, title, value, color=None):
        frame = ctk.CTkFrame(
            parent,
            fg_color=self.card_bg,
            corner_radius=14,
            border_width=1,
            border_color=self.card_border,
            height=92,
        )
        frame.grid_propagate(False)
        t_color = color if color else "gray"
        lbl_title = ctk.CTkLabel(frame, text=title.upper(), font=("Roboto", 11, "bold"), text_color=t_color)
        lbl_title.pack(pady=(10,0))
        
        # Use a smaller font if value is long
        font_size = 20
        if len(value) > 20: font_size = 12
        elif len(value) > 10: font_size = 14
        
        lbl_val = ctk.CTkLabel(
            frame,
            text=value,
            font=("Roboto", font_size, "bold"),
            wraplength=220,
            justify="center",
        )
        lbl_val.pack(pady=(2,10), padx=10, fill="both", expand=True)
        frame.value_label = lbl_val
        frame.title_label = lbl_title
        return frame

    def _add_tooltip(self, widget, text):
        DashboardToolTip(widget, text)
        for child in widget.winfo_children():
            DashboardToolTip(child, text)

    def _install_tooltips(self):
        self._add_tooltip(
            self.card_total_time,
            "Total lesson recording time in this profile, using each lesson's saved duration.",
        )
        self._add_tooltip(
            self.card_student_pct,
            "The share of total lesson time where your assigned student speaker was talking.",
        )
        self._add_tooltip(
            self.card_wpm,
            "Your average speaking speed: student words divided by student speaking minutes.",
        )
        self._add_tooltip(
            self.card_words,
            "Total words attributed to your assigned student speaker across all lessons.",
        )
        self._add_tooltip(
            self.card_grammar,
            "Average AI grammar score across lessons with completed AI analysis.",
        )
        self._add_tooltip(
            self.card_auto_gap,
            "Warm-session grammar average minus cold-session grammar average over the latest 10 analyzed lessons. "
            "Warm means at least 1 hour of practice in the prior 7 days; cold means less. "
            "A positive gap suggests you perform better after recent practice.",
        )
        self._add_tooltip(
            self.card_latency,
            "Average pause before you respond after another speaker, ignoring pauses over 10 seconds.",
        )
        self._add_tooltip(
            self.card_max_turn,
            "The longest single student speaking turn found in your lessons.",
        )
        self._add_tooltip(
            self.golden_panel,
            "Recent target vocabulary from completed AI analyses, deduplicated from newest lessons backward.",
        )
        DashboardToolTip(
            self.ai_btn,
            "Analyze the newest incomplete lessons only. Stops once it reaches an older lesson that already has complete AI metrics.",
        )
        DashboardToolTip(
            self.ai_backfill_btn,
            "Analyze every incomplete lesson in this profile, including older gaps. Use this to repair historical dashboard data.",
        )

    def run_ai_analysis(self):
        """Recomputes LLM analysis for the newest incomplete lessons only."""
        self._run_ai_analysis(select_pending_ai_lesson_dirs, "recent")

    def run_ai_backfill(self):
        """Recomputes LLM analysis for every incomplete lesson in the profile."""
        self._run_ai_analysis(select_all_incomplete_ai_lesson_dirs, "all")

    def _run_ai_analysis(self, selector, scope):
        if not self.pipeline:
            self.status_lbl.configure(text="Error: Pipeline not connected")
            return

        provider, model, api_key = self._dashboard_ai_settings()
        if provider == "openai" and not api_key:
            self.status_lbl.configure(text="OpenAI API key missing. Save it in the Analysis window first.")
            return

        self.ai_btn.configure(state="disabled", text="Computing...")
        self.ai_backfill_btn.configure(state="disabled", text="Backfilling...")
        
        def _thread_target():
            lessons_dir = os.path.join(self.profile_dir, "lessons")
            if not os.path.isdir(lessons_dir): return

            to_process = selector(lessons_dir)
            
            total = len(to_process)
            if total == 0:
                self.after(0, lambda: self._on_ai_finished(0, 0, scope))
                return

            # Process loop
            processed = 0
            for i, path in enumerate(to_process):
                msg = f"Analyzing {i+1}/{total}: {os.path.basename(path)}"
                self.after(0, lambda m=msg: self.status_lbl.configure(text=m))
                
                success = self.pipeline.compute_ai_metrics(path, model=model, mode=provider, api_key=api_key)
                if success:
                    processed += 1
                else:
                    print(f"Skipping lesson {path} due to AI error.")
            
            self.after(0, lambda: self._on_ai_finished(processed, total, scope))

        threading.Thread(target=_thread_target, daemon=True).start()

    def _on_ai_finished(self, count, total, scope="recent"):
        self.ai_btn.configure(state="normal", text="✨ Compute Recent AI Metrics")
        self.ai_backfill_btn.configure(state="normal", text="Backfill All AI Metrics")
        if total == 0:
            if scope == "all":
                self.status_lbl.configure(text="All lessons already have complete AI metrics.")
            else:
                self.status_lbl.configure(text="No recent analyzable lessons found.")
        else:
            self.status_lbl.configure(text=f"Finished analyzing {count} lessons.")
            self.refresh_data()

    def _dashboard_ai_settings(self):
        cfg_path = os.path.join(self.profile_dir, "config.json")
        cfg = {}
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f) or {}
            except Exception:
                cfg = {}

        provider = cfg.get("llm_provider", "ollama")
        if provider == "openai":
            model = cfg.get("openai_model", "gpt-5.4")
            api_key = deobfuscate_secret(cfg.get("openai_api_key", ""))
            return provider, model, api_key

        return "ollama", cfg.get("ollama_model", DEFAULT_OLLAMA_ANALYSIS_MODEL), None

    def _parse_lesson_datetime(self, lesson_id, meta):
        dt_obj = None
        if "recorded_at" in meta and meta["recorded_at"]:
            try: dt_obj = datetime.fromisoformat(meta["recorded_at"])
            except: pass
        if not dt_obj and "created_at" in meta:
            try: dt_obj = datetime.fromisoformat(meta["created_at"])
            except: pass
        if not dt_obj:
            try: dt_obj = datetime.strptime(lesson_id.split("_")[0], "%Y%m%d")
            except: pass
        return dt_obj

    def _practice_hours_by_lesson(self, lessons_dir):
        lesson_times = []
        for lesson_id in os.listdir(lessons_dir):
            path = os.path.join(lessons_dir, lesson_id)
            meta_path = os.path.join(path, "meta.json")
            if not (os.path.isdir(path) and os.path.isfile(meta_path)):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                dt_obj = self._parse_lesson_datetime(lesson_id, meta)
                duration_sec = float(meta.get("duration_sec", 0.0) or 0.0)
                if dt_obj:
                    lesson_times.append((lesson_id, dt_obj, max(0.0, duration_sec)))
            except Exception:
                continue

        practice_hours = {}
        for lesson_id, dt_obj, _duration_sec in lesson_times:
            window_start = dt_obj - timedelta(days=7)
            seconds = sum(
                duration_sec
                for other_id, other_dt, duration_sec in lesson_times
                if other_id != lesson_id and window_start <= other_dt < dt_obj
            )
            practice_hours[lesson_id] = seconds / 3600.0
        return practice_hours

    def refresh_data(self):
        if not self.profile_dir or not os.path.exists(self.profile_dir): return

        lessons_dir = os.path.join(self.profile_dir, "lessons")
        if not os.path.isdir(lessons_dir): return
        practice_hours_by_lesson = self._practice_hours_by_lesson(lessons_dir)

        # Accumulators
        total_recording_sec = 0.0
        student_speaking_sec = 0.0
        student_total_words = 0
        total_latency_sum = 0.0
        total_latency_count = 0
        max_turn_duration = 0.0
        
        # AI Accumulators
        all_grammar_scores = [] # list of (date, score)
        golden_words_all = []
        context_sessions = []
        context_trend = []
        
        student_words_by_month = defaultdict(int)
        fluency_trend = []
        has_data = False

        for lesson_id in os.listdir(lessons_dir):
            path = os.path.join(lessons_dir, lesson_id)
            meta_path = os.path.join(path, "meta.json")
            seg_path = os.path.join(path, "segments.json")
            ai_path = os.path.join(path, "ai_stats.json")

            if not (os.path.isdir(path) and os.path.isfile(meta_path) and os.path.isfile(seg_path)):
                continue

            try:
                with open(meta_path, 'r', encoding='utf-8') as f: meta = json.load(f)
                with open(seg_path, 'r', encoding='utf-8') as f: segments = json.load(f)
                
                # --- Date Parsing (Reused Logic) ---
                dt_obj = self._parse_lesson_datetime(lesson_id, meta)
                
                month_key = dt_obj.strftime("%Y-%m") if dt_obj else "Unknown"

                # --- AI Data Loading ---
                ai_data = {}
                if os.path.exists(ai_path):
                    try:
                        with open(ai_path, 'r', encoding='utf-8') as f: ai_data = json.load(f)
                        if "grammar_score" in ai_data and dt_obj:
                            score = ai_data["grammar_score"]
                            # Basic validation to ensure it's a number
                            if isinstance(score, (int, float)):
                                all_grammar_scores.append((dt_obj, score))
                        golden_words = ai_data.get("golden_words", [])
                        if isinstance(golden_words, str):
                            golden_words = [golden_words]
                        if isinstance(golden_words, list):
                            golden_words_all.extend(
                                str(w).strip() for w in golden_words if str(w).strip()
                            )
                    except: pass

                # --- Identity Logic ---
                student_ids = set()
                if "student_speakers" in meta: student_ids.update(meta["student_speakers"])
                
                # --- Standard Metrics ---
                lesson_dur = float(meta.get("duration_sec", 0.0))
                total_recording_sec += lesson_dur
                has_data = True

                lesson_student_words = 0
                lesson_student_sec = 0.0
                lesson_lat_sum = 0.0
                lesson_lat_cnt = 0
                last_end = 0.0
                last_was_student = False

                for i, seg in enumerate(segments):
                    spk = seg.get("speaker", "UNKNOWN")
                    start = float(seg.get("start", 0))
                    end = float(seg.get("end", 0))
                    dur = end - start
                    text = seg.get("text", "").strip()
                    
                    is_student = (spk in student_ids) or (not student_ids and "01" in spk) 

                    if is_student:
                        lesson_student_sec += dur
                        wc = len(text.split())
                        lesson_student_words += wc
                        student_total_words += wc
                        if dur > max_turn_duration: max_turn_duration = dur
                        
                        if i > 0 and not last_was_student:
                            lat = start - last_end
                            if 0.0 < lat < 10.0:
                                lesson_lat_sum += lat
                                lesson_lat_cnt += 1
                        last_was_student = True
                    else:
                        last_was_student = False
                    last_end = end

                student_words_by_month[month_key] += lesson_student_words
                total_latency_sum += lesson_lat_sum
                total_latency_count += lesson_lat_cnt
                student_speaking_sec += lesson_student_sec
                
                lesson_raw_wpm = (lesson_student_words / (lesson_student_sec/60)) if lesson_student_sec > 10 else None
                if lesson_raw_wpm is not None and dt_obj:
                    lat = (lesson_lat_sum / lesson_lat_cnt) if lesson_lat_cnt else 0
                    fluency_trend.append((dt_obj, lesson_raw_wpm, lat))

                if dt_obj:
                    context = ai_data.get("context_metrics") if isinstance(ai_data.get("context_metrics"), dict) else {}
                    derived_practice_hours = practice_hours_by_lesson.get(lesson_id)
                    context = build_context_metrics(
                        raw_grammar_score=ai_data.get("grammar_score", context.get("raw_grammar_score")),
                        raw_wpm=lesson_raw_wpm if lesson_raw_wpm is not None else context.get("raw_wpm"),
                        practice_hours_last_7_days=(
                            derived_practice_hours
                            if derived_practice_hours is not None
                            else context.get("practice_hours_last_7_days")
                        ),
                        topic_difficulty=ai_data.get("topic_difficulty", context.get("topic_difficulty")),
                        idea_density=ai_data.get("idea_density", context.get("idea_density")),
                        abstraction_level=ai_data.get("abstraction_level", context.get("abstraction_level")),
                        cognitive_branching=ai_data.get("cognitive_branching", context.get("cognitive_branching")),
                        technical_density=ai_data.get("technical_density", context.get("technical_density")),
                        discourse_depth=ai_data.get("discourse_depth", context.get("discourse_depth")),
                        lexical_retrieval_pressure=ai_data.get(
                            "lexical_retrieval_pressure",
                            context.get("lexical_retrieval_pressure"),
                        ),
                        fatigue_or_stress=context.get("fatigue_or_stress"),
                        long_pauses_per_min=context.get("long_pauses_per_min"),
                        self_repairs_per_min=context.get("self_repairs_per_min"),
                        filled_pauses_per_min=context.get("filled_pauses_per_min"),
                        notes=ai_data.get("context_notes", context.get("notes")),
                    )
                    context_sessions.append(
                        {
                            "date": dt_obj,
                            "grammar_score": ai_data.get("grammar_score"),
                            "context_metrics": context,
                        }
                    )
                    context_trend.append((dt_obj, context))

            except Exception as e:
                print(f"Skipping {lesson_id}: {e}")

        # --- UPDATE UI CARDS ---
        if not has_data:
            self.card_total_time.value_label.configure(text="0.0")
            return

        total_hours = total_recording_sec / 3600.0
        pct = (student_speaking_sec / total_recording_sec * 100) if total_recording_sec else 0
        wpm_global = (student_total_words / (student_speaking_sec/60)) if student_speaking_sec > 30 else 0
        avg_latency = (total_latency_sum / total_latency_count) if total_latency_count else 0.0
        
        self.card_total_time.value_label.configure(text=f"{total_hours:.1f}")
        self.card_student_pct.value_label.configure(text=f"{pct:.1f}%")
        self.card_wpm.value_label.configure(text=f"{wpm_global:.0f}")
        self.card_words.value_label.configure(text=f"{student_total_words:,}")
        self.card_latency.value_label.configure(text=f"{avg_latency:.2f}s")
        self.card_max_turn.value_label.configure(text=f"{max_turn_duration:.1f}s")

        # --- UPDATE AI CARDS ---
        if all_grammar_scores:
            # Calculate Average
            scores_only = [x[1] for x in all_grammar_scores]
            avg_gram = sum(scores_only) / len(scores_only)
            self.card_grammar.value_label.configure(text=f"{avg_gram:.0f}/100")
        else:
            self.card_grammar.value_label.configure(text="--")

        auto_gap = compute_automaticity_gap(sorted(context_sessions, key=lambda x: x["date"]), window=10)
        if auto_gap["automaticity_gap"] is not None:
            self.card_auto_gap.value_label.configure(
                text=(
                    f"{auto_gap['automaticity_gap']:.1f} pts\n"
                    f"W {auto_gap['warm_session_count']} / C {auto_gap['cold_session_count']}"
                )
            )
        else:
            self.card_auto_gap.value_label.configure(
                text=f"--\nW {auto_gap['warm_session_count']} / C {auto_gap['cold_session_count']}"
            )

        if golden_words_all:
            # Show last 3 unique words
            unique_gold = []
            seen = set()
            for w in reversed(golden_words_all):
                if w not in seen:
                    unique_gold.append(w)
                    seen.add(w)
                if len(unique_gold) >= 3: break

            for idx, lbl in enumerate(self.golden_word_labels):
                if idx < len(unique_gold):
                    lbl.configure(text=unique_gold[idx])
                else:
                    lbl.configure(text="--")
        else:
            for lbl in self.golden_word_labels:
                lbl.configure(text="No Analysis")

        # --- PLOTS (In Tabs) ---
        self._plot_activity(student_words_by_month, self.tab_activity)
        self._plot_fluency(fluency_trend, self.tab_fluency)
        self._plot_grammar(all_grammar_scores, self.tab_grammar)
        self._plot_context_metrics(context_trend, self.tab_context)

    # --- PLOT FUNCTIONS ---

    def _plot_activity(self, data, parent_tab):
        for widget in parent_tab.winfo_children(): widget.destroy()
        if not data: return

        sorted_keys = sorted(data.keys())
        values = [data[k] for k in sorted_keys]
        labels = []
        for k in sorted_keys:
            try: labels.append(datetime.strptime(k, "%Y-%m").strftime("%b"))
            except: labels.append(str(k))

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
        fig.patch.set_facecolor(self.bg_figure)
        ax.set_facecolor(self.bg_figure)
        
        ax.bar(labels, values, color=self.color_student)
        ax.set_title("Total Words Spoken (Monthly)", color=self.text_color, fontsize=10)
        ax.tick_params(colors=self.text_color, labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self.text_color)
        ax.spines['left'].set_color(self.text_color)

        canvas = FigureCanvasTkAgg(fig, master=parent_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_fluency(self, trend_data, parent_tab):
        for widget in parent_tab.winfo_children(): widget.destroy()
        if len(trend_data) < 2:
            ctk.CTkLabel(parent_tab, text="Need more lessons to show trend").pack(expand=True)
            return

        trend_data.sort(key=lambda x: x[0])
        dates = [x[0] for x in trend_data]
        wpms = [x[1] for x in trend_data]
        lats = [x[2] for x in trend_data]

        fig, ax1 = plt.subplots(figsize=(5, 3.5), dpi=100)
        fig.patch.set_facecolor(self.bg_figure)
        ax1.set_facecolor(self.bg_figure)

        color = self.color_student
        ax1.plot(dates, wpms, color=color, marker='o', label="WPM")
        ax1.set_ylabel("WPM", color=color, fontsize=9)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=9)
        ax1.tick_params(axis='x', colors=self.text_color, labelsize=9)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_color(self.text_color)
        ax1.spines['left'].set_color(self.text_color)
        
        ax2 = ax1.twinx() 
        color2 = self.color_bad 
        ax2.plot(dates, lats, color=color2, marker='x', linestyle='--', label="Lat")
        ax2.set_ylabel("Latency (s)", color=color2, fontsize=9)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_color(self.text_color)
        ax2.spines['bottom'].set_visible(False)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        canvas = FigureCanvasTkAgg(fig, master=parent_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_grammar(self, scores_data, parent_tab):
        for widget in parent_tab.winfo_children(): widget.destroy()
        if len(scores_data) < 2:
            ctk.CTkLabel(parent_tab, text="Compute AI metrics for more lessons to see trend.").pack(expand=True)
            return

        scores_data.sort(key=lambda x: x[0])
        dates = [x[0] for x in scores_data]
        scores = [x[1] for x in scores_data]

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
        fig.patch.set_facecolor(self.bg_figure)
        ax.set_facecolor(self.bg_figure)
        
        ax.plot(dates, scores, color=self.color_ai, marker='D', linewidth=2)
        ax.set_title("Grammar Accuracy Score (0-100)", color=self.text_color, fontsize=10)
        ax.set_ylim(0, 105) # Keep scale consistent
        
        ax.tick_params(colors=self.text_color, labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self.text_color)
        ax.spines['left'].set_color(self.text_color)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

        canvas = FigureCanvasTkAgg(fig, master=parent_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_context_metrics(self, trend_data, parent_tab):
        for widget in parent_tab.winfo_children(): widget.destroy()
        if len(trend_data) < 2:
            ctk.CTkLabel(parent_tab, text="Need more context metrics to show trend.").pack(expand=True)
            return

        trend_data.sort(key=lambda x: x[0])
        score_series = {
            "Raw Grammar": ("raw_grammar_score", self.color_ai, "o", "-", -0.30),
            "Adj Grammar": ("adjusted_grammar_score", "#6EB5FF", "D", "--", -0.18),
            "Raw WPM": ("raw_wpm", self.color_student, "o", "-", -0.06),
            "Load WPM": ("cognitive_load_adjusted_wpm", "#E0B84D", "s", "--", 0.06),
            "Fluency Load": ("fluency_under_load", "#64C2A6", "^", "-.", 0.18),
            "Effective Fluency": ("effective_fluency_score", "#E58ACD", "P", ":", 0.30),
            "Resilience": ("complexity_resilience_score", "#B6D957", "X", ":", 0.42),
        }
        advanced_series = {
            "Concept Load": ("conceptual_load_score", "#E0B84D", "o", "-", -0.18),
            "Abstraction": ("abstraction_level", "#6EB5FF", "D", "--", -0.10),
            "Branching": ("cognitive_branching", "#E58ACD", "s", "-.", -0.02),
            "Technical": ("technical_density", "#D76A5D", "^", "-", 0.06),
            "Discourse": ("discourse_depth", "#64C2A6", "P", "--", 0.14),
            "Lexical Pressure": ("lexical_retrieval_pressure", "#C98A1A", "x", ":", 0.22),
        }
        context_series = {
            "Topic Diff": ("topic_difficulty", "#D76A5D", "v", "-", -0.10),
            "Idea Density": ("idea_density", "#C98A1A", "P", "--", 0.00),
            "Practice 7d hrs": ("practice_hours_last_7_days", "#AFAFAF", "x", ":", 0.10),
        }

        fig, (ax_scores, ax_advanced, ax_context) = plt.subplots(
            3,
            1,
            figsize=(7.2, 5.6),
            dpi=100,
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.2, 1]},
        )
        fig.patch.set_facecolor(self.bg_figure)

        def style_axis(ax):
            ax.set_facecolor(self.bg_figure)
            ax.tick_params(colors=self.text_color, labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(self.text_color)
            ax.spines['left'].set_color(self.text_color)

        def plot_group(ax, series, zbase=10):
            plotted_any = False
            values_all = []
            for idx, (label, (key, color, marker, linestyle, day_offset)) in enumerate(series.items()):
                points = [(dt, ctx.get(key)) for dt, ctx in trend_data if isinstance(ctx.get(key), (int, float))]
                if len(points) < 2:
                    continue
                dates = [x[0] + timedelta(days=day_offset) for x in points]
                values = [x[1] for x in points]
                values_all.extend(values)
                ax.plot(
                    dates,
                    values,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.6,
                    markersize=4.3,
                    alpha=0.88,
                    label=label,
                    color=color,
                    zorder=zbase - idx,
                )
                plotted_any = True
            return plotted_any, values_all

        plotted_scores, _score_values = plot_group(ax_scores, score_series)
        plotted_advanced, advanced_values = plot_group(ax_advanced, advanced_series)
        plotted_context, context_values = plot_group(ax_context, context_series)

        if not (plotted_scores or plotted_advanced or plotted_context):
            plt.close(fig)
            ctk.CTkLabel(parent_tab, text="Context metrics are not available yet.").pack(expand=True)
            return

        for ax in (ax_scores, ax_advanced, ax_context):
            style_axis(ax)

        ax_scores.set_title("Scores and WPM", color=self.text_color, fontsize=10, pad=6)
        ax_scores.set_ylabel("Score / WPM", color=self.text_color, fontsize=8)
        if plotted_scores:
            ax_scores.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
                fontsize=7,
                framealpha=0.82,
            )
        else:
            ax_scores.text(0.5, 0.5, "No score/WPM context metrics yet", transform=ax_scores.transAxes, ha="center", color=self.card_subtext)

        ax_advanced.set_title("Advanced Conceptual Load", color=self.text_color, fontsize=10, pad=4)
        ax_advanced.set_ylabel("1-10", color=self.text_color, fontsize=8)
        if advanced_values:
            ax_advanced.set_ylim(0.5, max(10.0, max(advanced_values) + 0.75))
        if plotted_advanced:
            ax_advanced.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
                fontsize=7,
                framealpha=0.82,
            )
        else:
            ax_advanced.text(0.5, 0.5, "No advanced load metrics yet", transform=ax_advanced.transAxes, ha="center", color=self.card_subtext)

        ax_context.set_title("Context Inputs", color=self.text_color, fontsize=10, pad=4)
        ax_context.set_ylabel("Rating / hrs", color=self.text_color, fontsize=8)
        if context_values:
            context_top = max(5.0, max(context_values) + 0.75)
            ax_context.set_ylim(-0.25, context_top)
        if plotted_context:
            ax_context.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
                fontsize=7,
                framealpha=0.82,
            )
        else:
            ax_context.text(0.5, 0.5, "No context inputs yet", transform=ax_context.transAxes, ha="center", color=self.card_subtext)

        ax_context.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        fig.suptitle("Raw and Context-Adjusted Trends", color=self.text_color, fontsize=10)
        fig.tight_layout(rect=[0, 0, 0.86, 0.96])

        canvas = FigureCanvasTkAgg(fig, master=parent_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
