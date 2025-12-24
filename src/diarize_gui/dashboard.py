import os
import json
import calendar
from datetime import datetime
from collections import defaultdict
import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Use a safe backend for macOS/Windows
matplotlib.use("TkAgg")

class DashboardFrame(ctk.CTkFrame):
    def __init__(self, master, profile_name, profile_dir, **kwargs):
        super().__init__(master, **kwargs)
        self.profile_name = profile_name
        self.profile_dir = profile_dir
        
        # Colors suitable for Dark/Light mode
        self.color_primary = "#3B8ED0"
        self.color_secondary = "#E0E0E0"
        self.bg_figure = "#2b2b2b" if ctk.get_appearance_mode() == "Dark" else "#ffffff"
        self.text_color = "white" if ctk.get_appearance_mode() == "Dark" else "black"

        # Data placeholders
        self.lessons = []
        self.top_speakers = [] 
        
        # --- UI LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. KPI Cards Row
        self.kpi_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.kpi_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        self.card_hours = self._create_kpi_card(self.kpi_frame, "Total Hours", "0.0")
        self.card_words = self._create_kpi_card(self.kpi_frame, "Total Words", "0")
        self.card_streak = self._create_kpi_card(self.kpi_frame, "Lessons", "0")
        self.card_wpm = self._create_kpi_card(self.kpi_frame, "Avg WPM", "0")

        self.card_hours.pack(side="left", expand=True, fill="x", padx=5)
        self.card_words.pack(side="left", expand=True, fill="x", padx=5)
        self.card_streak.pack(side="left", expand=True, fill="x", padx=5)
        self.card_wpm.pack(side="left", expand=True, fill="x", padx=5)

        # 2. Charts Area
        self.charts_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.charts_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.charts_frame.grid_columnconfigure(0, weight=1)
        self.charts_frame.grid_columnconfigure(1, weight=1)

        # Left Chart: Activity History
        self.chart_frame_left = ctk.CTkFrame(self.charts_frame)
        self.chart_frame_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Right Chart: Speaker Balance
        self.chart_frame_right = ctk.CTkFrame(self.charts_frame)
        self.chart_frame_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # 3. Refresh Button
        self.refresh_btn = ctk.CTkButton(self, text="Refresh Stats", command=self.refresh_data)
        self.refresh_btn.grid(row=3, column=0, columnspan=2, pady=10)

        # Initial Load
        self.refresh_data()

    def _create_kpi_card(self, parent, title, value):
        frame = ctk.CTkFrame(parent)
        lbl_title = ctk.CTkLabel(frame, text=title.upper(), font=("Roboto", 12), text_color="gray")
        lbl_title.pack(pady=(10,0))
        lbl_val = ctk.CTkLabel(frame, text=value, font=("Roboto", 24, "bold"))
        lbl_val.pack(pady=(0,10))
        frame.value_label = lbl_val # Store reference to update later
        return frame

    def refresh_data(self):
        """Scans the profile directory and recalculates all metrics."""
        if not self.profile_dir or not os.path.exists(self.profile_dir):
            return

        lessons_dir = os.path.join(self.profile_dir, "lessons")
        if not os.path.isdir(lessons_dir):
            return

        self.lessons = []
        
        # Stats Aggregators
        total_sec = 0
        total_words = 0
        words_by_month = defaultdict(int)
        
        # Speaker Aggregators (Global)
        # Structure: speaker -> {'time': 0.0, 'words': 0, 'turns': 0}
        spk_stats = defaultdict(lambda: {'time': 0.0, 'words': 0, 'turns': 0})

        for lesson_id in os.listdir(lessons_dir):
            path = os.path.join(lessons_dir, lesson_id)
            meta_path = os.path.join(path, "meta.json")
            seg_path = os.path.join(path, "segments.json")

            if os.path.isdir(path) and os.path.isfile(meta_path) and os.path.isfile(seg_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    with open(seg_path, 'r', encoding='utf-8') as f:
                        segs = json.load(f)
                    
                    # Basic Lesson Stats
                    dt_str = meta.get("created_at", "")
                    if dt_str:
                        dt = datetime.fromisoformat(dt_str)
                        month_key = dt.strftime("%Y-%m") # YYYY-MM
                    else:
                        month_key = "Unknown"

                    total_sec += meta.get("duration_sec", 0)

                    # Detailed Segment Stats
                    for s in segs:
                        dur = s['end'] - s['start']
                        txt = s.get('text', "").strip()
                        wc = len(txt.split())
                        spk = s.get('speaker', "UNKNOWN")

                        total_words += wc
                        words_by_month[month_key] += wc
                        
                        spk_stats[spk]['time'] += dur
                        spk_stats[spk]['words'] += wc
                        spk_stats[spk]['turns'] += 1

                    self.lessons.append(meta)

                except Exception as e:
                    print(f"Error loading lesson {lesson_id}: {e}")

        # --- UPDATE KPIs ---
        self.card_hours.value_label.configure(text=f"{total_sec / 3600:.1f}")
        self.card_words.value_label.configure(text=f"{total_words:,}")
        self.card_streak.value_label.configure(text=str(len(self.lessons)))
        
        # Calculate Dominant Speaker WPM (Assuming Student is top speaker for now)
        # Or better: Average WPM across all speech
        if total_sec > 0:
             # Just a rough global WPM (total words / total recording minutes)
             # A more accurate one would be (student words / student minutes)
             avg_wpm = (total_words / (total_sec/60)) if total_sec else 0
             self.card_wpm.value_label.configure(text=f"{avg_wpm:.0f}")

        # --- UPDATE PLOTS ---
        self._plot_activity(words_by_month)
        self._plot_balance(spk_stats)

    def _plot_activity(self, data):
        # Clear previous
        for widget in self.chart_frame_left.winfo_children():
            widget.destroy()

        # Prepare Data
        sorted_keys = sorted(data.keys())
        values = [data[k] for k in sorted_keys]
        # Simplify labels (e.g. "Jan", "Feb")
        labels = []
        for k in sorted_keys:
            try:
                dt = datetime.strptime(k, "%Y-%m")
                labels.append(dt.strftime("%b"))
            except:
                labels.append(k)

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        fig.patch.set_facecolor(self.bg_figure)
        ax.set_facecolor(self.bg_figure)
        
        bars = ax.bar(labels, values, color=self.color_primary)
        
        ax.set_title("Words Spoken (History)", color=self.text_color, fontsize=10)
        ax.tick_params(axis='x', colors=self.text_color, labelsize=8)
        ax.tick_params(axis='y', colors=self.text_color, labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self.text_color)
        ax.spines['left'].set_color(self.text_color)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame_left)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_balance(self, spk_data):
        for widget in self.chart_frame_right.winfo_children():
            widget.destroy()

        if not spk_data:
            return

        # Sort speakers by time
        sorted_spks = sorted(spk_data.items(), key=lambda x: x[1]['time'], reverse=True)
        # Take top 3 max to avoid clutter
        top_spks = sorted_spks[:3]
        
        labels = [x[0] for x in top_spks]
        sizes = [x[1]['time'] / 60 for x in top_spks] # minutes

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        fig.patch.set_facecolor(self.bg_figure)
        
        # Pie Chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            textprops=dict(color=self.text_color),
            colors=["#3B8ED0", "#E0E0E0", "#FFB74D"]
        )
        
        ax.set_title("Speaking Time (Min)", color=self.text_color, fontsize=10)
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame_right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)