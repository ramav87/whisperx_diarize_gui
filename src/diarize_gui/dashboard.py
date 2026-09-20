import os
import json
import re
import threading
import queue
from datetime import date, datetime, timedelta
from collections import Counter, defaultdict
from urllib.parse import quote
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import requests
from tkinter import messagebox
from .theme import AppTheme
from .pipeline import DEFAULT_OLLAMA_ANALYSIS_MODEL
from .lesson_selection import select_all_incomplete_ai_lesson_dirs, select_pending_ai_lesson_dirs
from .metrics.context_adjusted import build_context_metrics, compute_automaticity_gap
from .metrics.language_growth import compute_language_growth_metrics
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
    def __init__(self, master, profile_name, profile_dir, pipeline=None, server_url=None, **kwargs):
        super().__init__(master, **kwargs)
        self.profile_name = str(profile_name) if profile_name else "Student"
        self.profile_dir = profile_dir
        self.pipeline = pipeline 
        self.server_url = server_url.rstrip("/") if isinstance(server_url, str) and server_url.strip() else None
        self.current_lesson_dir = None
        self._ui_queue = queue.SimpleQueue()
        self.goal_hours = 50.0
        self.goal_deadline = "2027-12-31"
        self._load_goal_settings()
        
        # Colors
        self.color_primary = AppTheme.BTN_PRIMARY
        self.color_student = AppTheme.BTN_SUCCESS
        self.color_tutor = "#C98A1A"
        self.color_bad    = "#D76A5D"
        self.color_ai     = "#9A4BCF"
        self.text_color = AppTheme.TEXT_PRIMARY
        self.card_bg = AppTheme.BG_CARD
        self.card_border = AppTheme.BORDER_DIVIDER
        self.card_subtext = AppTheme.TEXT_MUTED

        self.filler_pattern = re.compile(r"\b(um|uh|eh|mm|hm|este|em)\b", re.IGNORECASE)

        # --- UI LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.configure(fg_color="transparent")
        self.grid_rowconfigure(3, weight=1)

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
            text="SERVER PROFILE" if self.server_url else "LOCAL PROFILE",
            font=("Roboto", 10, "bold"),
            text_color=AppTheme.BTN_TEXT_ON_BLUE,
            fg_color=self.color_primary,
            corner_radius=999,
            padx=10,
            pady=4,
        )
        self.profile_pill.grid(row=0, column=1, rowspan=2, sticky="e")

        # Keep the complete snapshot in one horizontal strip on desktop. The old
        # two-row layout spent nearly 200 vertical pixels before reaching a chart.
        self.metrics_row = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 6))

        self.card_total_time = self._create_kpi_card(self.metrics_row, "Total Hours", "0.0")
        self.card_student_pct = self._create_kpi_card(self.metrics_row, "You Spoke", "0%")
        self.card_wpm = self._create_kpi_card(self.metrics_row, "Your WPM", "0")
        self.card_words = self._create_kpi_card(self.metrics_row, "Total Words", "0")
        self.card_grammar = self._create_kpi_card(self.metrics_row, "Avg Grammar", "--", color=self.color_ai)
        self.card_auto_gap = self._create_kpi_card(self.metrics_row, "Auto. Gap", "--", color=self.color_ai)
        self.card_latency = self._create_kpi_card(self.metrics_row, "Avg Latency", "0.0s")
        self.card_max_turn = self._create_kpi_card(self.metrics_row, "Longest Turn", "0s")

        metric_cards = (
            self.card_total_time,
            self.card_student_pct,
            self.card_wpm,
            self.card_words,
            self.card_grammar,
            self.card_auto_gap,
            self.card_latency,
            self.card_max_turn,
        )
        for column, card in enumerate(metric_cards):
            self.metrics_row.grid_columnconfigure(column, weight=1, uniform="dashboard_metric")
            card.grid(row=0, column=column, sticky="ew", padx=4)

        # Golden words panel
        self.golden_panel = ctk.CTkFrame(
            self,
            fg_color=self.card_bg,
            corner_radius=14,
            border_width=1,
            border_color=self.card_border,
        )
        self.golden_panel.grid(row=2, column=1, sticky="nsew", padx=(6, 12), pady=(0, 6))
        self.golden_panel.grid_columnconfigure(1, weight=1)

        self.golden_header = ctk.CTkLabel(
            self.golden_panel,
            text="Recent Golden Words",
            font=("Roboto", 12, "bold"),
            text_color=self.color_ai,
            anchor="w",
        )
        self.golden_header.grid(row=0, column=0, sticky="w", padx=(14, 8), pady=(9, 1))

        self.golden_hint = ctk.CTkLabel(
            self.golden_panel,
            text="Latest vocabulary targets pulled from AI analysis.",
            font=("Roboto", 11),
            text_color=self.card_subtext,
            anchor="w",
        )
        self.golden_hint.grid(row=1, column=0, sticky="w", padx=(14, 8), pady=(0, 9))

        self.golden_words_container = ctk.CTkFrame(self.golden_panel, fg_color="transparent")
        self.golden_words_container.grid(row=0, column=1, rowspan=2, sticky="ew", padx=(8, 14), pady=8)
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
                pady=6,
                justify="center",
                anchor="center",
                wraplength=220,
            )
            lbl.grid(row=0, column=idx, sticky="ew", padx=4)
            self.golden_word_labels.append(lbl)

        self.goal_panel = ctk.CTkFrame(
            self,
            fg_color=self.card_bg,
            corner_radius=14,
            border_width=1,
            border_color=self.card_border,
        )
        self.goal_panel.grid(row=2, column=0, sticky="nsew", padx=(12, 6), pady=(0, 6))
        self.goal_panel.grid_columnconfigure(0, weight=1)
        goal_header = ctk.CTkFrame(self.goal_panel, fg_color="transparent")
        goal_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 0))
        goal_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            goal_header,
            text="Speaking Goal",
            font=("Roboto", 12, "bold"),
            text_color=self.color_student,
        ).grid(row=0, column=0, sticky="w")
        self.goal_edit_btn = ctk.CTkButton(
            goal_header,
            text="Edit",
            width=58,
            height=26,
            fg_color="transparent",
            border_width=1,
            border_color=self.card_border,
            command=self._open_goal_editor,
        )
        self.goal_edit_btn.grid(row=0, column=1, sticky="e")
        self.goal_value_label = ctk.CTkLabel(
            self.goal_panel,
            text="0.0 / 50 hours",
            font=("Roboto", 17, "bold"),
            text_color=self.text_color,
            anchor="w",
        )
        self.goal_value_label.grid(row=1, column=0, sticky="ew", padx=14, pady=(1, 1))
        self.goal_progress = ctk.CTkProgressBar(
            self.goal_panel,
            height=8,
            progress_color=self.color_student,
        )
        self.goal_progress.grid(row=2, column=0, sticky="ew", padx=14)
        self.goal_progress.set(0)
        self.goal_detail_label = ctk.CTkLabel(
            self.goal_panel,
            text="Set a target to calculate your weekly pace.",
            font=("Roboto", 10),
            text_color=self.card_subtext,
            anchor="w",
        )
        self.goal_detail_label.grid(row=3, column=0, sticky="ew", padx=14, pady=(3, 8))

        # 3. Charts Area (Now Tabbed!)
        self.chart_tabs = ctk.CTkTabview(self)
        self.chart_tabs.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 4))
        
        self.tab_activity = self.chart_tabs.add("Activity")
        self.tab_fluency = self.chart_tabs.add("Fluency")
        self.tab_grammar = self.chart_tabs.add("Grammar AI")
        self.tab_growth = self.chart_tabs.add("Language Growth")
        self.tab_vocabulary = self.chart_tabs.add("Vocabulary")
        self.tab_context = self.chart_tabs.add("Context (Experimental)")
        
        # 4. Controls Row
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 8))

        self.refresh_btn = ctk.CTkButton(
            self.controls_frame,
            text="Refresh",
            command=self.refresh_data,
            fg_color=self.color_primary,
            hover_color=AppTheme.BTN_PRIMARY_HOVER,
        )
        self.refresh_btn.pack(side="left", padx=(0, 6))

        self.ai_btn = ctk.CTkButton(
            self.controls_frame,
            text="Analyze Latest",
            fg_color=self.color_ai,
            hover_color="#7B1FA2",
            command=self.run_ai_analysis,
        )
        self.ai_btn.pack(side="left", padx=6)

        self.ai_backfill_btn = ctk.CTkButton(
            self.controls_frame,
            text="Backfill AI History",
            fg_color="#7B1FA2",
            hover_color="#5E167A",
            command=self.run_ai_backfill,
        )
        self.ai_backfill_btn.pack(side="left", padx=6)
        
        self.status_lbl = ctk.CTkLabel(self.controls_frame, text="", text_color=self.card_subtext)
        self.status_lbl.pack(side="left", padx=10, fill="x", expand=True)

        self._install_tooltips()
        self.after(25, self._drain_ui_queue)

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

    def set_server_url(self, server_url):
        self.server_url = server_url.rstrip("/") if isinstance(server_url, str) and server_url.strip() else None
        self.profile_pill.configure(text="SERVER PROFILE" if self.server_url else "LOCAL PROFILE")

    def _load_goal_settings(self):
        """Load speaking-goal preferences, preferring the active server profile."""
        settings = {}
        config_path = os.path.join(self.profile_dir, "config.json") if self.profile_dir else None
        if config_path and os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    local_config = json.load(handle) or {}
                if isinstance(local_config, dict):
                    settings.update(local_config)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if self.server_url:
            try:
                profile = quote(self.profile_name, safe="")
                response = requests.get(f"{self.server_url}/api/profiles/{profile}", timeout=5)
                response.raise_for_status()
                body = response.json()
                remote = body.get("settings") if isinstance(body, dict) else None
                if isinstance(remote, dict):
                    settings.update(remote)
            except (requests.RequestException, TypeError, ValueError):
                pass
        try:
            hours = float(settings.get("speaking_goal_hours", self.goal_hours))
            if hours > 0:
                self.goal_hours = hours
        except (TypeError, ValueError):
            pass
        deadline = str(settings.get("speaking_goal_deadline", self.goal_deadline))
        try:
            datetime.strptime(deadline, "%Y-%m-%d")
            self.goal_deadline = deadline
        except ValueError:
            pass

    def _save_goal_settings(self):
        """Persist the speaking goal locally and to the active server profile."""
        values = {
            "speaking_goal_hours": self.goal_hours,
            "speaking_goal_deadline": self.goal_deadline,
        }
        if self.profile_dir:
            os.makedirs(self.profile_dir, exist_ok=True)
            config_path = os.path.join(self.profile_dir, "config.json")
            config = {}
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as handle:
                        config = json.load(handle) or {}
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    config = {}
            if not isinstance(config, dict):
                config = {}
            config.update(values)
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
        if self.server_url:
            profile = quote(self.profile_name, safe="")
            response = requests.patch(
                f"{self.server_url}/api/profiles/{profile}",
                json={"settings": values},
                timeout=10,
            )
            response.raise_for_status()

    def _open_goal_editor(self):
        """Open a compact editor for target hours and completion date."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Edit speaking goal")
        dialog.geometry("390x245")
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        ctk.CTkLabel(dialog, text="Speaking Goal", font=("Roboto", 20, "bold")).pack(pady=(18, 10))
        body = ctk.CTkFrame(dialog, fg_color="transparent")
        body.pack(fill="x", padx=24)
        ctk.CTkLabel(body, text="Target learner speaking hours").grid(row=0, column=0, sticky="w", pady=5)
        hours_entry = ctk.CTkEntry(body, width=120)
        hours_entry.insert(0, f"{self.goal_hours:g}")
        hours_entry.grid(row=0, column=1, sticky="e", pady=5)
        ctk.CTkLabel(body, text="Deadline (YYYY-MM-DD)").grid(row=1, column=0, sticky="w", pady=5)
        deadline_entry = ctk.CTkEntry(body, width=120)
        deadline_entry.insert(0, self.goal_deadline)
        deadline_entry.grid(row=1, column=1, sticky="e", pady=5)
        body.grid_columnconfigure(0, weight=1)

        def save():
            try:
                hours = float(hours_entry.get())
                if hours <= 0:
                    raise ValueError("Target hours must be greater than zero.")
                deadline = deadline_entry.get().strip()
                datetime.strptime(deadline, "%Y-%m-%d")
                self.goal_hours = hours
                self.goal_deadline = deadline
                self._save_goal_settings()
                self.refresh_data()
                dialog.destroy()
            except ValueError as exc:
                messagebox.showerror("Invalid speaking goal", str(exc), parent=dialog)
            except (OSError, requests.RequestException) as exc:
                messagebox.showerror("Could not save goal", str(exc), parent=dialog)

        ctk.CTkButton(dialog, text="Save Goal", command=save).pack(pady=18)

    def _render_goal_progress(self, summary):
        spoken_hours = max(0.0, self._number_or_zero(summary.get("student_speaking_sec")) / 3600.0)
        target = max(0.01, self.goal_hours)
        remaining = max(0.0, target - spoken_hours)
        completion = min(1.0, spoken_hours / target)
        self.goal_progress.set(completion)
        self.goal_value_label.configure(text=f"{spoken_hours:.1f} / {target:g} hours ({completion:.0%})")
        deadline = datetime.strptime(self.goal_deadline, "%Y-%m-%d").date()
        days_left = max(0, (deadline - date.today()).days)
        if remaining <= 0:
            detail = f"Goal reached · target date {deadline.strftime('%b %Y')}"
        elif days_left == 0:
            detail = f"{remaining:.1f} hours remaining · deadline reached"
        else:
            minutes_per_week = remaining * 60.0 / (days_left / 7.0)
            detail = (
                f"{remaining:.1f} hours remaining · {minutes_per_week:.0f} min/week "
                f"to {deadline.strftime('%b %Y')}"
            )
        self.goal_detail_label.configure(text=detail)

    def _create_kpi_card(self, parent, title, value, color=None):
        frame = ctk.CTkFrame(
            parent,
            fg_color=self.card_bg,
            corner_radius=14,
            border_width=1,
            border_color=self.card_border,
            height=76,
        )
        frame.grid_propagate(False)
        t_color = color if color else "gray"
        lbl_title = ctk.CTkLabel(frame, text=title.upper(), font=("Roboto", 10, "bold"), text_color=t_color)
        lbl_title.pack(pady=(7, 0))
        
        # Use a smaller font if value is long
        font_size = 20
        if len(value) > 20: font_size = 12
        elif len(value) > 10: font_size = 14
        
        lbl_val = ctk.CTkLabel(
            frame,
            text=value,
            font=("Roboto", min(font_size, 18), "bold"),
            wraplength=220,
            justify="center",
        )
        lbl_val.pack(pady=(1, 7), padx=6, fill="both", expand=True)
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
        self._add_tooltip(
            self.goal_panel,
            "Progress uses only time attributed to your learner speaker, not the full lesson duration.",
        )
        DashboardToolTip(
            self.goal_edit_btn,
            "Change your target speaking hours and deadline.",
        )
        DashboardToolTip(
            self.refresh_btn,
            "Reload lesson files and redraw every dashboard metric and chart.",
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
            if self.server_url:
                self._run_server_ai_analysis(provider, model, api_key, scope)
                return

            lessons_dir = os.path.join(self.profile_dir, "lessons")
            if not os.path.isdir(lessons_dir): return

            to_process = selector(lessons_dir)
            
            total = len(to_process)
            if total == 0:
                self._post_ui(lambda: self._on_ai_finished(0, 0, scope))
                return

            # Process loop
            processed = 0
            for i, path in enumerate(to_process):
                msg = f"Analyzing {i+1}/{total}: {os.path.basename(path)}"
                self._post_ui(lambda m=msg: self.status_lbl.configure(text=m))
                
                success = self.pipeline.compute_ai_metrics(path, model=model, mode=provider, api_key=api_key)
                if success:
                    processed += 1
                else:
                    print(f"Skipping lesson {path} due to AI error.")
            
            self._post_ui(lambda: self._on_ai_finished(processed, total, scope))

        threading.Thread(target=_thread_target, daemon=True).start()

    def _run_server_ai_analysis(self, provider, model, api_key, scope):
        """Backfill the authoritative server lessons used by the dashboard."""
        try:
            profile = quote(self.profile_name, safe="")
            response = requests.get(
                f"{self.server_url}/api/profiles/{profile}/dashboard",
                timeout=60,
            )
            response.raise_for_status()
            lessons = response.json().get("lessons") or []
            candidates = []
            for lesson in lessons:
                if not isinstance(lesson, dict) or not lesson.get("id"):
                    continue
                if lesson.get("analysis_complete"):
                    if scope == "recent":
                        break
                    continue
                candidates.append(lesson)

            total = len(candidates)
            if total == 0:
                self._post_ui(lambda: self._on_ai_finished(0, 0, scope))
                return

            processed = 0
            for index, lesson in enumerate(candidates, start=1):
                lesson_id = str(lesson["id"])
                message = f"Analyzing server lesson {index}/{total}: {lesson_id}"
                self._post_ui(lambda text=message: self.status_lbl.configure(text=text))
                payload = {"provider": provider, "model": model}
                if api_key:
                    payload["api_key"] = api_key
                result = requests.post(
                    f"{self.server_url}/api/lessons/{quote(lesson_id, safe='')}/analyze",
                    json=payload,
                    timeout=900,
                )
                if result.ok:
                    processed += 1
                    body = result.json()
                    self._mirror_server_ai_stats(lesson_id, body.get("ai_stats"))
                else:
                    try:
                        detail = result.json().get("detail")
                    except (TypeError, ValueError):
                        detail = result.text.strip()
                    print(f"Skipping server lesson {lesson_id}: {detail or result.status_code}")

            self._post_ui(lambda: self._on_ai_finished(processed, total, scope))
        except Exception as exc:
            message = f"Server AI backfill failed: {exc}"
            print(message)
            self._post_ui(lambda text=message: self._on_ai_failed(text))

    def _post_ui(self, callback):
        self._ui_queue.put(callback)

    def _drain_ui_queue(self):
        try:
            while True:
                self._ui_queue.get_nowait()()
        except queue.Empty:
            pass
        except Exception as exc:
            print(f"[WARN] Dashboard UI callback failed: {exc}")
        try:
            self.after(25, self._drain_ui_queue)
        except Exception:
            pass

    def _mirror_server_ai_stats(self, lesson_id, ai_stats):
        """Keep existing desktop lesson mirrors consistent with server analysis."""
        if not isinstance(ai_stats, dict):
            return
        lessons_dir = os.path.join(self.profile_dir, "lessons")
        if not os.path.isdir(lessons_dir):
            return
        for local_id in os.listdir(lessons_dir):
            lesson_dir = os.path.join(lessons_dir, local_id)
            if not os.path.isdir(lesson_dir):
                continue
            meta = {}
            meta_path = os.path.join(lesson_dir, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as handle:
                        meta = json.load(handle) or {}
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    meta = {}
            if local_id == lesson_id or meta.get("server_lesson_id") == lesson_id:
                with open(os.path.join(lesson_dir, "ai_stats.json"), "w", encoding="utf-8") as handle:
                    json.dump(ai_stats, handle, ensure_ascii=False, indent=2)

    def _on_ai_failed(self, message):
        self.ai_btn.configure(state="normal", text="✨ Compute Recent AI Metrics")
        self.ai_backfill_btn.configure(state="normal", text="Backfill All AI Metrics")
        self.status_lbl.configure(text=message)

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
            except (TypeError, ValueError): pass
        if not dt_obj and "created_at" in meta:
            try: dt_obj = datetime.fromisoformat(meta["created_at"])
            except (TypeError, ValueError): pass
        if not dt_obj:
            try: dt_obj = datetime.strptime(lesson_id.split("_")[0], "%Y%m%d")
            except (TypeError, ValueError): pass
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
        if self._refresh_from_server_dashboard():
            return
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
        language_growth_trend = []
        vocabulary_by_month = defaultdict(Counter)
        
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
                    except (OSError, TypeError, ValueError, json.JSONDecodeError): pass

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

                language_metrics = compute_language_growth_metrics(
                    segments,
                    meta.get("student_speakers") or [],
                )
                if dt_obj and language_metrics.get("token_count"):
                    language_growth_trend.append((dt_obj, language_metrics))
                    for item in language_metrics.get("top_content_words", []):
                        if isinstance(item, dict) and item.get("word"):
                            vocabulary_by_month[month_key][str(item["word"])] += int(item.get("count") or 0)

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

        summary = {
            "lesson_count": len(os.listdir(lessons_dir)) if os.path.isdir(lessons_dir) else 0,
            "total_hours": total_recording_sec / 3600.0,
            "student_speaking_sec": student_speaking_sec,
            "student_speaking_pct": (student_speaking_sec / total_recording_sec * 100) if total_recording_sec else 0,
            "global_wpm": (student_total_words / (student_speaking_sec / 60)) if student_speaking_sec > 30 else 0,
            "student_total_words": student_total_words,
            "avg_latency_sec": (total_latency_sum / total_latency_count) if total_latency_count else None,
            "max_turn_duration_sec": max_turn_duration,
            "average_grammar_score": None,
            "automaticity_gap": compute_automaticity_gap(sorted(context_sessions, key=lambda x: x["date"]), window=10),
            "golden_words": [],
        }

        if not has_data:
            self.card_total_time.value_label.configure(text="0.0")
            return
        if all_grammar_scores:
            scores_only = [x[1] for x in all_grammar_scores]
            summary["average_grammar_score"] = sum(scores_only) / len(scores_only)
        if golden_words_all:
            unique_gold = []
            seen = set()
            for w in reversed(golden_words_all):
                if w not in seen:
                    unique_gold.append(w)
                    seen.add(w)
                if len(unique_gold) >= 3: break
            summary["golden_words"] = unique_gold

        payload = {
            "summary": summary,
            "trends": {
                "activity": [{"month": month, "student_words": words} for month, words in student_words_by_month.items()],
                "fluency": [
                    {"date": dt_obj.isoformat(), "raw_wpm": wpm, "avg_latency_sec": latency}
                    for dt_obj, wpm, latency in fluency_trend
                ],
                "grammar": [
                    {"date": dt_obj.isoformat(), "grammar_score": score}
                    for dt_obj, score in all_grammar_scores
                ],
                "context": [
                    {"date": dt_obj.isoformat(), "context_metrics": context}
                    for dt_obj, context in context_trend
                ],
                "language_growth": [
                    {
                        "date": dt_obj.isoformat(),
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key not in {"top_content_words", "limitations"}
                        },
                    }
                    for dt_obj, metrics in language_growth_trend
                ],
                "vocabulary": [
                    {
                        "month": month,
                        "words": [
                            {"word": word, "count": count}
                            for word, count in counts.most_common(40)
                        ],
                    }
                    for month, counts in sorted(vocabulary_by_month.items())
                ],
            },
        }
        self._render_dashboard_payload(payload)

    def _refresh_from_server_dashboard(self):
        if not self.server_url or not self.profile_name:
            return False
        try:
            profile = quote(self.profile_name, safe="")
            response = requests.get(f"{self.server_url}/api/profiles/{profile}/dashboard", timeout=10)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return False
            self.profile_pill.configure(text="SERVER PROFILE")
            self.status_lbl.configure(text="")
            self._render_dashboard_payload(payload)
            return True
        except Exception as e:
            print(f"[WARN] Could not refresh dashboard from server: {e}")
            self.profile_pill.configure(text="LOCAL FALLBACK")
            return False

    def _render_dashboard_payload(self, payload):
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        trends = payload.get("trends") if isinstance(payload.get("trends"), dict) else {}

        total_hours = self._number_or_zero(summary.get("total_hours"))
        pct = self._number_or_zero(summary.get("student_speaking_pct"))
        wpm_global = self._number_or_zero(summary.get("global_wpm"))
        student_total_words = int(self._number_or_zero(summary.get("student_total_words")))
        avg_latency = self._number_or_none(summary.get("avg_latency_sec"))
        max_turn_duration = self._number_or_zero(summary.get("max_turn_duration_sec"))

        self._render_goal_progress(summary)

        self.card_total_time.value_label.configure(text=f"{total_hours:.1f}")
        self.card_student_pct.value_label.configure(text=f"{pct:.1f}%")
        self.card_wpm.value_label.configure(text=f"{wpm_global:.0f}")
        self.card_words.value_label.configure(text=f"{student_total_words:,}")
        self.card_latency.value_label.configure(text=f"{avg_latency:.2f}s" if avg_latency is not None else "--")
        self.card_max_turn.value_label.configure(text=f"{max_turn_duration:.1f}s")

        avg_grammar = self._number_or_none(summary.get("average_grammar_score"))
        self.card_grammar.value_label.configure(text=f"{avg_grammar:.0f}/100" if avg_grammar is not None else "--")

        auto_gap = summary.get("automaticity_gap") if isinstance(summary.get("automaticity_gap"), dict) else {}
        warm_count = int(self._number_or_zero(auto_gap.get("warm_session_count")))
        cold_count = int(self._number_or_zero(auto_gap.get("cold_session_count")))
        gap = self._number_or_none(auto_gap.get("automaticity_gap"))
        if gap is not None:
            self.card_auto_gap.value_label.configure(text=f"{gap:.1f} pts\nW {warm_count} / C {cold_count}")
        else:
            self.card_auto_gap.value_label.configure(text=f"--\nW {warm_count} / C {cold_count}")

        golden_words = summary.get("golden_words") if isinstance(summary.get("golden_words"), list) else []
        for idx, lbl in enumerate(self.golden_word_labels):
            if idx < len(golden_words):
                lbl.configure(text=str(golden_words[idx]))
            else:
                lbl.configure(text="No Analysis" if not golden_words else "--")

        student_words_by_month = {
            str(item.get("month")): int(self._number_or_zero(item.get("student_words")))
        for item in (trends.get("activity") or [])
            if isinstance(item, dict) and item.get("month")
        }
        fluency_trend = []
        for item in (trends.get("fluency") or []):
            if not isinstance(item, dict):
                continue
            dt_obj = self._parse_iso_datetime(item.get("date"))
            raw_wpm = self._number_or_none(item.get("raw_wpm"))
            if dt_obj and raw_wpm is not None:
                fluency_trend.append((dt_obj, raw_wpm, self._number_or_none(item.get("avg_latency_sec"))))
        grammar_scores = []
        for item in (trends.get("grammar") or []):
            if not isinstance(item, dict):
                continue
            dt_obj = self._parse_iso_datetime(item.get("date"))
            score = self._number_or_none(item.get("grammar_score"))
            if dt_obj and score is not None:
                grammar_scores.append((dt_obj, score))
        context_trend = []
        for item in (trends.get("context") or []):
            if not isinstance(item, dict):
                continue
            dt_obj = self._parse_iso_datetime(item.get("date"))
            context = item.get("context_metrics")
            if dt_obj and isinstance(context, dict):
                context_trend.append((dt_obj, context))
        language_growth_trend = []
        for item in (trends.get("language_growth") or []):
            if not isinstance(item, dict):
                continue
            dt_obj = self._parse_iso_datetime(item.get("date"))
            if dt_obj:
                language_growth_trend.append((dt_obj, item))
        vocabulary_by_month = {
            str(item.get("month")): item.get("words") or []
            for item in (trends.get("vocabulary") or [])
            if isinstance(item, dict) and item.get("month")
        }

        legacy_count = int(self._number_or_zero(summary.get("legacy_ai_metrics_count")))
        if legacy_count:
            self.status_lbl.configure(
                text=f"{legacy_count} legacy AI analyses need Backfill for full-transcript metrics."
            )

        month_dates = []
        for month in student_words_by_month:
            try:
                month_dates.append(datetime.strptime(month, "%Y-%m"))
            except (TypeError, ValueError):
                continue
        all_chart_dates = (
            month_dates
            + [item[0] for item in fluency_trend]
            + [item[0] for item in grammar_scores]
            + [item[0] for item in context_trend]
            + [item[0] for item in language_growth_trend]
        )
        self._chart_date_bounds = (
            (min(all_chart_dates), max(all_chart_dates)) if all_chart_dates else None
        )

        self._plot_activity(student_words_by_month, self.tab_activity)
        self._plot_fluency(fluency_trend, self.tab_fluency)
        self._plot_grammar(grammar_scores, self.tab_grammar)
        self._plot_language_growth(language_growth_trend, self.tab_growth)
        self._plot_vocabulary(vocabulary_by_month, self.tab_vocabulary)
        self._plot_context_metrics(context_trend, self.tab_context)

    def _parse_iso_datetime(self, value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    def _number_or_none(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _number_or_zero(self, value):
        parsed = self._number_or_none(value)
        return parsed if parsed is not None else 0.0

    # --- PLOT FUNCTIONS ---

    def _clear_chart(self, parent_tab):
        previous_figure = getattr(parent_tab, "_dashboard_figure", None)
        if previous_figure is not None:
            plt.close(previous_figure)
        for widget in parent_tab.winfo_children():
            widget.destroy()

    def _empty_chart(self, parent_tab, message):
        self._clear_chart(parent_tab)
        ctk.CTkLabel(
            parent_tab,
            text=message,
            text_color=self.card_subtext,
            font=("Roboto", 13),
        ).pack(expand=True)

    def _new_chart(self, *, rows=1, height=3.6, sharex=False, height_ratios=None):
        figure, axes = plt.subplots(
            rows,
            1,
            figsize=(9.5, height),
            dpi=100,
            sharex=sharex,
            gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
        )
        figure.patch.set_facecolor(self.card_bg)
        return figure, axes

    def _style_axis(self, axis, *, title=None, ylabel=None, show_x=True, title_pad=12):
        axis.set_facecolor(self.card_bg)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=self.card_border, linewidth=0.8, alpha=0.72)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(
            axis="both",
            colors=self.card_subtext,
            labelsize=9,
            length=0,
            pad=7,
        )
        if not show_x:
            axis.tick_params(axis="x", labelbottom=False)
        if title:
            axis.set_title(
                title,
                loc="left",
                color=self.text_color,
                fontsize=12,
                fontweight="bold",
                pad=title_pad,
            )
        if ylabel:
            axis.set_ylabel(ylabel, color=self.card_subtext, fontsize=9, labelpad=8)

    def _style_date_axis(self, axis, dates):
        dates = sorted(date for date in dates if isinstance(date, datetime))
        bounds = getattr(self, "_chart_date_bounds", None)
        if bounds:
            start, end = bounds
        elif dates:
            start, end = dates[0], dates[-1]
        else:
            return

        if start == end:
            start -= timedelta(days=16)
            end += timedelta(days=16)
        else:
            padding = max(timedelta(days=8), (end - start) * 0.04)
            start -= padding
            end += padding

        axis.set_xlim(start, end)
        axis.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=2, maxticks=7, interval_multiples=True)
        )
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    def _style_legend(self, axis, **kwargs):
        legend = axis.legend(
            frameon=False,
            fontsize=8,
            labelcolor=self.text_color,
            **kwargs,
        )
        return legend

    def _register_legend_toggles(self, figure, legend, lines_by_label):
        """Make legend entries act as visibility switches for dense charts."""
        pick_map = getattr(figure, "_dashboard_legend_pick_map", {})
        for legend_line, legend_text in zip(legend.get_lines(), legend.get_texts()):
            data_line = lines_by_label.get(legend_text.get_text())
            if data_line is None:
                continue
            legend_line.set_picker(7)
            legend_text.set_picker(True)
            target = (data_line, legend_line, legend_text)
            pick_map[legend_line] = target
            pick_map[legend_text] = target
        figure._dashboard_legend_pick_map = pick_map

    def _embed_chart(
        self,
        figure,
        parent_tab,
        *,
        right=0.97,
        top=0.88,
        bottom=0.18,
        hspace=0.34,
    ):
        # Keep titles, legends, tick labels, and figure captions inside the
        # Tk canvas at every window size. Matplotlib's default margins assume
        # a standalone window and are too tight inside a CTkTabview.
        figure.subplots_adjust(
            left=0.10,
            right=right,
            bottom=bottom,
            top=top,
            hspace=hspace,
        )
        parent_tab._dashboard_figure = figure
        canvas = FigureCanvasTkAgg(figure, master=parent_tab)

        pick_map = getattr(figure, "_dashboard_legend_pick_map", {})
        if pick_map:
            def toggle_series(event):
                target = pick_map.get(event.artist)
                if target is None:
                    return
                data_line, legend_line, legend_text = target
                visible = not data_line.get_visible()
                data_line.set_visible(visible)
                alpha = 1.0 if visible else 0.28
                legend_line.set_alpha(alpha)
                legend_text.set_alpha(alpha)
                canvas.draw_idle()

            canvas.mpl_connect("pick_event", toggle_series)

        canvas.draw()
        canvas.get_tk_widget().configure(highlightthickness=0, background=self.card_bg)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_activity(self, data, parent_tab):
        self._clear_chart(parent_tab)
        points = []
        for month, value in data.items():
            try:
                points.append((datetime.strptime(month, "%Y-%m"), value))
            except (TypeError, ValueError):
                continue
        if not points:
            self._empty_chart(parent_tab, "Process a lesson to begin the activity timeline.")
            return

        points.sort(key=lambda item: item[0])
        dates = [item[0] for item in points]
        values = [item[1] for item in points]
        figure, axis = self._new_chart()
        axis.bar(
            dates,
            values,
            width=20,
            color=self.color_student,
            alpha=0.88,
            edgecolor="none",
        )
        self._style_axis(axis, title="Monthly speaking volume", ylabel="Student words")
        self._style_date_axis(axis, dates)
        self._embed_chart(figure, parent_tab)

    def _plot_fluency(self, trend_data, parent_tab):
        self._clear_chart(parent_tab)
        if len(trend_data) < 2:
            self._empty_chart(parent_tab, "At least two lessons are needed for a fluency trend.")
            return

        trend_data.sort(key=lambda x: x[0])
        dates = [x[0] for x in trend_data]
        wpms = [x[1] for x in trend_data]
        latency_points = [(x[0], x[2]) for x in trend_data if x[2] is not None]

        figure, wpm_axis = self._new_chart()
        wpm_axis.plot(
            dates,
            wpms,
            color=self.color_student,
            marker="o",
            markersize=5,
            linewidth=2.4,
            label="Speaking pace",
        )
        wpm_axis.fill_between(dates, wpms, alpha=0.08, color=self.color_student)
        self._style_axis(wpm_axis, title="Fluency over time", ylabel="Words / min")

        latency_axis = None
        if latency_points:
            latency_axis = wpm_axis.twinx()
            latency_axis.plot(
                [point[0] for point in latency_points],
                [point[1] for point in latency_points],
                color=self.color_bad,
                marker="o",
                markersize=4,
                linewidth=1.8,
                linestyle="--",
                label="Response latency",
            )
            for spine in latency_axis.spines.values():
                spine.set_visible(False)
            latency_axis.tick_params(
                axis="y", colors=self.color_bad, labelsize=9, length=0, pad=7
            )
            latency_axis.set_ylabel("Latency (sec)", color=self.color_bad, fontsize=9, labelpad=8)
        self._style_date_axis(wpm_axis, dates)
        handles1, labels1 = wpm_axis.get_legend_handles_labels()
        handles2, labels2 = latency_axis.get_legend_handles_labels() if latency_axis else ([], [])
        wpm_axis.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            borderaxespad=0,
            ncol=2,
            frameon=False,
            fontsize=8,
            labelcolor=self.text_color,
        )
        self._embed_chart(figure, parent_tab, right=0.92, top=0.82)

    def _plot_grammar(self, scores_data, parent_tab):
        self._clear_chart(parent_tab)
        if len(scores_data) < 2:
            self._empty_chart(parent_tab, "Analyze at least two lessons to show a grammar trend.")
            return

        scores_data.sort(key=lambda x: x[0])
        dates = [x[0] for x in scores_data]
        scores = [x[1] for x in scores_data]

        figure, axis = self._new_chart()
        axis.plot(
            dates,
            scores,
            color=self.color_ai,
            marker="o",
            markersize=5,
            linewidth=2.4,
        )
        axis.fill_between(dates, scores, alpha=0.09, color=self.color_ai)
        axis.set_ylim(0, 105)
        self._style_axis(axis, title="Grammar accuracy", ylabel="Score / 100")
        self._style_date_axis(axis, dates)
        self._embed_chart(figure, parent_tab)

    def _plot_language_growth(self, trend_data, parent_tab):
        self._clear_chart(parent_tab)
        if len(trend_data) < 2:
            self._empty_chart(parent_tab, "At least two learner-scoped lessons are needed for growth trends.")
            return

        trend_data = sorted(trend_data, key=lambda item: item[0])
        figure, (lexical_axis, structure_axis) = self._new_chart(
            rows=2,
            height=4.8,
            sharex=True,
            height_ratios=[1, 1],
        )

        def plot_metric(axis, key, label, color, linestyle="-"):
            points = [
                (date, metrics.get(key))
                for date, metrics in trend_data
                if isinstance(metrics.get(key), (int, float))
            ]
            if len(points) < 2:
                return None
            line, = axis.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.8,
                linestyle=linestyle,
                label=label,
            )
            return line

        lexical_line = plot_metric(
            lexical_axis, "mattr_50", "Lexical diversity (MATTR-50)", self.color_ai
        )
        lexical_axis.set_ylim(0, 1)
        self._style_axis(
            lexical_axis,
            title="Lexical range",
            ylabel="Diversity (0–1)",
            show_x=False,
        )
        # The single-series label belongs in the heading, not over the data.
        if lexical_line:
            lexical_axis.set_title(
                "Lexical range  ·  MATTR-50",
                loc="left",
                color=self.text_color,
                fontsize=12,
                fontweight="bold",
                pad=12,
            )

        structure_lines = {}
        for key, label, color, linestyle in (
            ("mean_utterance_words", "Mean utterance length", self.color_student, "-"),
            ("connectors_per_100_words", "Connectors / 100 words", "#6EB5FF", "--"),
            ("subordinators_per_100_words", "Subordinators / 100 words", "#E0B84D", ":"),
        ):
            line = plot_metric(structure_axis, key, label, color, linestyle)
            if line:
                structure_lines[label] = line
        self._style_axis(
            structure_axis,
            title="Utterance and clause complexity",
            ylabel="Words / rate",
            title_pad=36,
        )
        self._style_date_axis(structure_axis, [item[0] for item in trend_data])
        if structure_lines:
            legend = self._style_legend(
                structure_axis,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                borderaxespad=0,
                ncol=3,
            )
            self._register_legend_toggles(figure, legend, structure_lines)

        figure.text(
            0.99,
            0.025,
            "Descriptive transcript measures; click a legend item to toggle",
            ha="right",
            color=self.card_subtext,
            fontsize=8,
        )
        self._embed_chart(
            figure,
            parent_tab,
            top=0.88,
            bottom=0.20,
            hspace=0.70,
        )

    def _plot_vocabulary(self, vocabulary_by_month, parent_tab):
        self._clear_chart(parent_tab)
        if not vocabulary_by_month:
            self._empty_chart(parent_tab, "No learner vocabulary is available yet.")
            return

        months = sorted(vocabulary_by_month)
        labels = {
            datetime.strptime(month, "%Y-%m").strftime("%b '%y"): month
            for month in months
        }
        controls = ctk.CTkFrame(parent_tab, fg_color="transparent")
        controls.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(
            controls,
            text="Month",
            text_color=self.card_subtext,
        ).pack(side="left", padx=(0, 8))
        selected = ctk.StringVar(value=list(labels)[-1])
        plot_host = ctk.CTkFrame(parent_tab, fg_color="transparent")
        plot_host.pack(fill="both", expand=True)

        def render_month(label):
            self._render_vocabulary_month(
                plot_host,
                labels[label],
                vocabulary_by_month.get(labels[label], []),
            )

        menu = ctk.CTkOptionMenu(
            controls,
            variable=selected,
            values=list(labels),
            command=render_month,
            width=130,
        )
        menu.pack(side="left")
        ctk.CTkLabel(
            controls,
            text="Frequent learner word forms; stopwords and fillers excluded",
            text_color=self.card_subtext,
        ).pack(side="left", padx=12)
        render_month(selected.get())

    def _render_vocabulary_month(self, parent, month, words):
        self._clear_chart(parent)
        cleaned = [
            (str(item.get("word")), int(item.get("count") or 0))
            for item in words
            if isinstance(item, dict) and item.get("word") and int(item.get("count") or 0) > 0
        ][:30]
        if not cleaned:
            self._empty_chart(parent, "No vocabulary terms were found for this month.")
            return

        figure, axis = self._new_chart(height=3.7)
        axis.set_facecolor(self.card_bg)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        maximum = max(count for _, count in cleaned)
        minimum = min(count for _, count in cleaned)
        colors = [self.color_ai, self.color_student, "#6EB5FF", "#E0B84D", "#E58ACD"]
        columns = 6
        rows = 5
        for index, (word, count) in enumerate(cleaned[: columns * rows]):
            row, column = divmod(index, columns)
            x = (column + 0.5) / columns
            y = 0.88 - row * (0.75 / max(1, rows - 1))
            weight = (count - minimum) / max(1, maximum - minimum)
            axis.text(
                x,
                y,
                word,
                ha="center",
                va="center",
                fontsize=10 + weight * 17,
                fontweight="bold" if weight > 0.65 else "normal",
                color=colors[index % len(colors)],
                alpha=0.78 + weight * 0.22,
                transform=axis.transAxes,
            )
        title = datetime.strptime(month, "%Y-%m").strftime("Vocabulary used in %b '%y")
        axis.set_title(title, loc="left", color=self.text_color, fontsize=12, fontweight="bold")
        self._embed_chart(figure, parent, top=0.90)

    def _plot_context_metrics(self, trend_data, parent_tab):
        self._clear_chart(parent_tab)
        if len(trend_data) < 2:
            self._empty_chart(parent_tab, "At least two analyzed lessons are needed for context trends.")
            return

        trend_data.sort(key=lambda x: x[0])
        score_series = {
            "Raw Grammar": ("raw_grammar_score", self.color_ai, "o", "-"),
            "Adj Grammar": ("adjusted_grammar_score", "#6EB5FF", "o", "--"),
            "Raw WPM": ("raw_wpm", self.color_student, "o", "-"),
            "Load WPM": ("cognitive_load_adjusted_wpm", "#E0B84D", "o", "--"),
            "Fluency Load": ("fluency_under_load", "#64C2A6", "o", "-."),
            "Effective Fluency": ("effective_fluency_score", "#E58ACD", "o", ":"),
            "Resilience": ("complexity_resilience_score", "#B6D957", "o", ":"),
        }
        advanced_series = {
            "Concept Load": ("conceptual_load_score", "#E0B84D", "o", "-"),
            "Abstraction": ("abstraction_level", "#6EB5FF", "o", "--"),
            "Branching": ("cognitive_branching", "#E58ACD", "o", "-."),
            "Technical": ("technical_density", "#D76A5D", "o", "-"),
            "Discourse": ("discourse_depth", "#64C2A6", "o", "--"),
            "Lexical Pressure": ("lexical_retrieval_pressure", "#C98A1A", "o", ":"),
        }
        context_series = {
            "Topic Diff": ("topic_difficulty", "#D76A5D", "o", "-"),
            "Idea Density": ("idea_density", "#C98A1A", "o", "--"),
            "Practice 7d hrs": ("practice_hours_last_7_days", "#AFAFAF", "o", ":"),
        }

        fig, (ax_scores, ax_advanced, ax_context) = self._new_chart(
            rows=3,
            height=5.2,
            sharex=True,
            height_ratios=[1.25, 1.0, 1.0],
        )

        def plot_group(ax, series, zbase=10):
            plotted_any = False
            values_all = []
            lines_by_label = {}
            for idx, (label, (key, color, marker, linestyle)) in enumerate(series.items()):
                points = [(dt, ctx.get(key)) for dt, ctx in trend_data if isinstance(ctx.get(key), (int, float))]
                if len(points) < 2:
                    continue
                dates = [x[0] for x in points]
                values = [x[1] for x in points]
                values_all.extend(values)
                line, = ax.plot(
                    dates,
                    values,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.8,
                    markersize=3.8,
                    alpha=0.9,
                    label=label,
                    color=color,
                    zorder=zbase - idx,
                )
                lines_by_label[label] = line
                plotted_any = True
            return plotted_any, values_all, lines_by_label

        plotted_scores, _score_values, score_lines = plot_group(ax_scores, score_series)
        plotted_advanced, advanced_values, advanced_lines = plot_group(ax_advanced, advanced_series)
        plotted_context, context_values, context_lines = plot_group(ax_context, context_series)

        if not (plotted_scores or plotted_advanced or plotted_context):
            plt.close(fig)
            ctk.CTkLabel(parent_tab, text="Context metrics are not available yet.").pack(expand=True)
            return

        self._style_axis(ax_scores, title="Experimental performance indices", ylabel="Score / WPM", show_x=False)
        self._style_axis(ax_advanced, title="Conceptual load", ylabel="Level", show_x=False)
        self._style_axis(ax_context, title="Lesson context", ylabel="Rating / hrs")
        self._style_date_axis(ax_context, [item[0] for item in trend_data])

        if plotted_scores:
            legend = self._style_legend(
                ax_scores,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
            )
            self._register_legend_toggles(fig, legend, score_lines)
        else:
            ax_scores.text(0.5, 0.5, "No score/WPM context metrics yet", transform=ax_scores.transAxes, ha="center", color=self.card_subtext)

        if advanced_values:
            ax_advanced.set_ylim(0.5, max(10.0, max(advanced_values) + 0.75))
        if plotted_advanced:
            legend = self._style_legend(
                ax_advanced,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
            )
            self._register_legend_toggles(fig, legend, advanced_lines)
        else:
            ax_advanced.text(0.5, 0.5, "No advanced load metrics yet", transform=ax_advanced.transAxes, ha="center", color=self.card_subtext)

        if context_values:
            context_top = max(5.0, max(context_values) + 0.75)
            ax_context.set_ylim(-0.25, context_top)
        if plotted_context:
            legend = self._style_legend(
                ax_context,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
            )
            self._register_legend_toggles(fig, legend, context_lines)
        else:
            ax_context.text(0.5, 0.5, "No context inputs yet", transform=ax_context.transAxes, ha="center", color=self.card_subtext)

        fig.text(
            0.99,
            0.025,
            "Click a legend item to show or hide its line",
            ha="right",
            color=self.card_subtext,
            fontsize=8,
        )
        self._embed_chart(
            fig,
            parent_tab,
            right=0.76,
            top=0.88,
            bottom=0.18,
            hspace=0.48,
        )
