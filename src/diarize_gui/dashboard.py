import os
import json
import re
import threading
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
    def __init__(self, master, profile_name, profile_dir, pipeline=None, **kwargs):
        super().__init__(master, **kwargs)
        self.profile_name = str(profile_name) if profile_name else "Student"
        self.profile_dir = profile_dir
        self.pipeline = pipeline # Access to backend for AI calls
        
        # Colors
        self.color_primary = "#3B8ED0"     
        self.color_student = "#4CAF50"     
        self.color_tutor = "#FF9800"       
        self.color_bad    = "#E57373"      
        self.color_ai     = "#9C27B0"      # Purple for AI stats
        self.bg_figure = "#2b2b2b" if ctk.get_appearance_mode() == "Dark" else "#ffffff"
        self.text_color = "white" if ctk.get_appearance_mode() == "Dark" else "black"

        self.filler_pattern = re.compile(r"\b(um|uh|eh|mm|hm|este|em)\b", re.IGNORECASE)

        # --- UI LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1) 

        # 1. Standard KPI Row
        self.row1 = ctk.CTkFrame(self, fg_color="transparent")
        self.row1.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
        
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
        self.row2.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))
        
        self.card_grammar = self._create_kpi_card(self.row2, "Grammar Score", "--", color=self.color_ai)
        self.card_golden = self._create_kpi_card(self.row2, "Golden Words", "--", color=self.color_ai)
        self.card_latency = self._create_kpi_card(self.row2, "Avg Latency", "0.0s")
        self.card_max_turn = self._create_kpi_card(self.row2, "Longest Turn", "0s")

        self.card_grammar.pack(side="left", expand=True, fill="x", padx=5)
        self.card_golden.pack(side="left", expand=True, fill="x", padx=5)
        self.card_latency.pack(side="left", expand=True, fill="x", padx=5)
        self.card_max_turn.pack(side="left", expand=True, fill="x", padx=5)

        # 3. Charts Area
        self.charts_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.charts_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.charts_frame.grid_columnconfigure(0, weight=1)
        self.charts_frame.grid_columnconfigure(1, weight=1)

        self.chart_frame_left = ctk.CTkFrame(self.charts_frame)
        self.chart_frame_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.chart_frame_right = ctk.CTkFrame(self.charts_frame)
        self.chart_frame_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # 4. Controls Row
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=3, column=0, columnspan=2, pady=10)

        self.refresh_btn = ctk.CTkButton(self.controls_frame, text="Refresh Data", command=self.refresh_data)
        self.refresh_btn.pack(side="left", padx=10)

        self.ai_btn = ctk.CTkButton(self.controls_frame, text="✨ Compute AI Metrics", 
                                    fg_color=self.color_ai, hover_color="#7B1FA2",
                                    command=self.run_ai_analysis)
        self.ai_btn.pack(side="left", padx=10)
        
        self.status_lbl = ctk.CTkLabel(self.controls_frame, text="", text_color="gray")
        self.status_lbl.pack(side="left", padx=10)

        # Initial Load
        self.refresh_data()

    def _create_kpi_card(self, parent, title, value, color=None):
        frame = ctk.CTkFrame(parent)
        t_color = color if color else "gray"
        lbl_title = ctk.CTkLabel(frame, text=title.upper(), font=("Roboto", 11), text_color=t_color)
        lbl_title.pack(pady=(8,0))
        
        # Use a smaller font if value is long (like a list of words)
        font_size = 20
        if len(value) > 15: font_size = 12
        
        lbl_val = ctk.CTkLabel(frame, text=value, font=("Roboto", font_size, "bold"))
        lbl_val.pack(pady=(0,8))
        frame.value_label = lbl_val
        return frame

    def run_ai_analysis(self):
        """Spawns a thread to run LLM analysis on all lessons missing ai_stats."""
        if not self.pipeline:
            self.status_lbl.configure(text="Error: Pipeline not connected")
            return
            
        self.ai_btn.configure(state="disabled", text="Computing...")
        self.status_lbl.configure(text="Analyzing latest lesson with LLM...")
        
        def _thread_target():
            lessons_dir = os.path.join(self.profile_dir, "lessons")
            if not os.path.isdir(lessons_dir): return
            
            # Find most recent lesson without stats, or all? 
            # Let's just do the most recent one for speed/demo purposes
            all_lessons = sorted(os.listdir(lessons_dir), reverse=True)
            
            count = 0
            for lid in all_lessons:
                path = os.path.join(lessons_dir, lid)
                if not os.path.isdir(path): continue
                
                stats_path = os.path.join(path, "ai_stats.json")
                if not os.path.exists(stats_path):
                    # Found a lesson needing analysis
                    success = self.pipeline.compute_ai_metrics(path, model="gemma:2b")
                    if success: count += 1
                    # Stop after 1 to avoid freezing app for minutes if user has 50 lessons
                    break 
            
            self.after(0, lambda: self._on_ai_finished(count))

        threading.Thread(target=_thread_target, daemon=True).start()

    def _on_ai_finished(self, count):
        self.ai_btn.configure(state="normal", text="✨ Compute AI Metrics")
        if count > 0:
            self.status_lbl.configure(text=f"Analyzed {count} lesson(s)!")
            self.refresh_data()
        else:
            self.status_lbl.configure(text="No new lessons to analyze.")

    def refresh_data(self):
        if not self.profile_dir or not os.path.exists(self.profile_dir): return

        lessons_dir = os.path.join(self.profile_dir, "lessons")
        if not os.path.isdir(lessons_dir): return

        # Accumulators
        total_recording_sec = 0.0
        student_speaking_sec = 0.0
        student_total_words = 0
        
        total_latency_sum = 0.0
        total_latency_count = 0
        max_turn_duration = 0.0
        
        # AI Accumulators
        grammar_scores = []
        golden_words_all = []
        
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
                
                # --- AI Data Loading ---
                if os.path.exists(ai_path):
                    try:
                        with open(ai_path, 'r', encoding='utf-8') as f: ai_data = json.load(f)
                        if "grammar_score" in ai_data: grammar_scores.append(ai_data["grammar_score"])
                        if "golden_words" in ai_data: golden_words_all.extend(ai_data["golden_words"])
                    except: pass

                # --- Identity Logic ---
                student_ids = set()
                if "student_speakers" in meta: student_ids.update(meta["student_speakers"])
                # Fallback logic could go here if needed...
                
                # --- Metrics ---
                lesson_dur = float(meta.get("duration_sec", 0.0))
                total_recording_sec += lesson_dur
                
                # --- UPDATED DATE PARSING LOGIC ---
                dt_obj = None
                
                # 1. Priority: 'recorded_at' (The actual lesson time)
                if "recorded_at" in meta and meta["recorded_at"]:
                    try: dt_obj = datetime.fromisoformat(meta["recorded_at"])
                    except: pass
                
                # 2. Fallback: 'created_at' (When file was made)
                if not dt_obj and "created_at" in meta and meta["created_at"]:
                    try: dt_obj = datetime.fromisoformat(meta["created_at"])
                    except: pass
                
                # 3. Last Resort: Folder Name (timestamp format)
                if not dt_obj:
                    try: dt_obj = datetime.strptime(lesson_id.split("_")[0], "%Y%m%d")
                    except: pass
                
                month_key = dt_obj.strftime("%Y-%m") if dt_obj else "Unknown"
                has_data = True

                lesson_student_words = 0
                lesson_student_sec = 0.0
                lesson_latency_sum = 0.0
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
                                lesson_latency_sum += lat
                                lesson_lat_cnt += 1
                        last_was_student = True
                    else:
                        last_was_student = False
                    last_end = end

                student_words_by_month[month_key] += lesson_student_words
                total_latency_sum += lesson_latency_sum
                total_latency_count += lesson_lat_cnt
                student_speaking_sec += lesson_student_sec
                
                if lesson_student_sec > 10 and dt_obj:
                    wpm = (lesson_student_words / (lesson_student_sec/60))
                    lat = (lesson_latency_sum / lesson_lat_cnt) if lesson_lat_cnt else 0
                    fluency_trend.append((dt_obj, wpm, lat))

            except Exception as e:
                print(f"Skipping {lesson_id}: {e}")

        # --- UPDATE UI ---
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
        if grammar_scores:
            avg_gram = sum(grammar_scores) / len(grammar_scores)
            self.card_grammar.value_label.configure(text=f"{avg_gram:.0f}/100")
        else:
            self.card_grammar.value_label.configure(text="--")

        if golden_words_all:
            # Show last 3 unique words (assuming simple strings)
            # Reverse list to show newest first, then take 3
            unique_gold = []
            seen = set()
            for w in reversed(golden_words_all):
                if w not in seen:
                    unique_gold.append(w)
                    seen.add(w)
                if len(unique_gold) >= 3: break
            
            text_gold = "\n".join(unique_gold)
            self.card_golden.value_label.configure(text=text_gold, font=("Roboto", 12))
        else:
            self.card_golden.value_label.configure(text="No Analysis")

        # Plots
        self._plot_activity(student_words_by_month)
        self._plot_fluency(fluency_trend)

    def _plot_activity(self, data):
        for widget in self.chart_frame_left.winfo_children(): widget.destroy()
        if not data: return

        sorted_keys = sorted(data.keys())
        values = [data[k] for k in sorted_keys]
        labels = []
        for k in sorted_keys:
            try: labels.append(datetime.strptime(k, "%Y-%m").strftime("%b"))
            except: labels.append(str(k))

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        fig.patch.set_facecolor(self.bg_figure)
        ax.set_facecolor(self.bg_figure)
        
        ax.bar(labels, values, color=self.color_student)
        ax.set_title("Volume (Words)", color=self.text_color, fontsize=10)
        ax.tick_params(colors=self.text_color, labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self.text_color)
        ax.spines['left'].set_color(self.text_color)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame_left)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_fluency(self, trend_data):
        for widget in self.chart_frame_right.winfo_children(): widget.destroy()
        if len(trend_data) < 2:
            ctk.CTkLabel(self.chart_frame_right, text="Need more data").pack(expand=True)
            return

        trend_data.sort(key=lambda x: x[0])
        dates = [x[0] for x in trend_data]
        wpms = [x[1] for x in trend_data]
        lats = [x[2] for x in trend_data]

        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=100)
        fig.patch.set_facecolor(self.bg_figure)
        ax1.set_facecolor(self.bg_figure)

        color = self.color_student
        ax1.set_title("Fluency Trend", color=self.text_color, fontsize=10)
        ax1.plot(dates, wpms, color=color, marker='o', label="WPM")
        ax1.set_ylabel("WPM", color=color, fontsize=8)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=8)
        ax1.tick_params(axis='x', colors=self.text_color, labelsize=8)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_color(self.text_color)
        ax1.spines['left'].set_color(self.text_color)
        
        ax2 = ax1.twinx() 
        color2 = self.color_bad 
        ax2.plot(dates, lats, color=color2, marker='x', linestyle='--', label="Lat")
        ax2.set_ylabel("Latency (s)", color=color2, fontsize=8)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_color(self.text_color)

        import matplotlib.dates as mdates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame_right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)