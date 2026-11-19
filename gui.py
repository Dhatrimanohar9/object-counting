"""
Graphical User Interface (GUI) Launcher for Vehicle Detection & Counting System.
Built with Tkinter and TTK for intuitive video selection, parameter tuning,
and real-time execution control.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Ensure current directory is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig, DEFAULT_VIDEO_PATH
from main import VehicleCountingSystem


class VehicleCountingGUI:
    """Tkinter-based control panel and parameter configurator."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Vehicle Detection, Tracking & Counting System")
        self.root.geometry("620x560")
        self.root.minsize(580, 520)

        self.system_thread = None
        self.current_system = None

        self._apply_styling()
        self._build_widgets()

    def _apply_styling(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 13, "bold"), foreground="#1e293b")
        style.configure("Section.TLabel", font=("Helvetica", 10, "bold"), foreground="#334155")
        style.configure("Action.TButton", font=("Helvetica", 10, "bold"), padding=6)
        style.configure("Count.TLabel", font=("Helvetica", 12, "bold"))

    def _build_widgets(self):
        main_frame = ttk.Frame(self.root, padding="16 16 16 16")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Title Banner
        title = ttk.Label(main_frame, text="Intelligent Vehicle Counting System", style="Header.TLabel")
        title.pack(anchor=tk.W, pady=(0, 4))
        subtitle = ttk.Label(
            main_frame,
            text="Kalman Filter Multi-Object Tracking & Bidirectional Analysis",
            foreground="#64748b"
        )
        subtitle.pack(anchor=tk.W, pady=(0, 14))

        # 2. Video Source Group
        video_group = ttk.LabelFrame(main_frame, text="Video Source Selection", padding="10 10 10 10")
        video_group.pack(fill=tk.X, pady=(0, 10))

        self.video_path_var = tk.StringVar(value=DEFAULT_VIDEO_PATH)
        path_entry = ttk.Entry(video_group, textvariable=self.video_path_var, width=46)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        browse_btn = ttk.Button(video_group, text="Browse...", command=self._browse_video)
        browse_btn.pack(side=tk.RIGHT)

        # 3. Parameters Group
        param_group = ttk.LabelFrame(main_frame, text="Detection & Tracking Parameters", padding="10 10 10 10")
        param_group.pack(fill=tk.X, pady=(0, 10))

        # Background subtraction method
        row1 = ttk.Frame(param_group)
        row1.pack(fill=tk.X, pady=4)
        ttk.Label(row1, text="Subtraction Method:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="MOG2")
        method_combo = ttk.Combobox(row1, textvariable=self.method_var, values=["MOG2", "KNN"], state="readonly", width=12)
        method_combo.pack(side=tk.RIGHT)

        # Virtual line position slider
        row2 = ttk.Frame(param_group)
        row2.pack(fill=tk.X, pady=4)
        ttk.Label(row2, text="Counting Line Position (Y %):").pack(side=tk.LEFT)
        self.line_pos_var = tk.DoubleVar(value=0.55)
        self.line_lbl = ttk.Label(row2, text="55%", width=6)
        self.line_lbl.pack(side=tk.RIGHT)
        line_slider = ttk.Scale(
            row2, from_=0.20, to_=0.80, variable=self.line_pos_var,
            command=lambda v: self.line_lbl.config(text=f"{int(float(v)*100)}%")
        )
        line_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8)

        # Min contour area
        row3 = ttk.Frame(param_group)
        row3.pack(fill=tk.X, pady=4)
        ttk.Label(row3, text="Min Contour Area (px):").pack(side=tk.LEFT)
        self.min_area_var = tk.IntVar(value=600)
        self.area_lbl = ttk.Label(row3, text="600", width=6)
        self.area_lbl.pack(side=tk.RIGHT)
        area_slider = ttk.Scale(
            row3, from_=200, to_=2500, variable=self.min_area_var,
            command=lambda v: self.area_lbl.config(text=str(int(float(v))))
        )
        area_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8)

        # Save output video checkbox
        self.save_video_var = tk.BooleanVar(value=True)
        save_chk = ttk.Checkbutton(param_group, text="Record and save processed output video", variable=self.save_video_var)
        save_chk.pack(anchor=tk.W, pady=(6, 2))

        # 4. Live Statistics Dashboard Card
        stats_group = ttk.LabelFrame(main_frame, text="Live Counting Statistics", padding="10 10 10 10")
        stats_group.pack(fill=tk.X, pady=(0, 12))

        stats_inner = ttk.Frame(stats_group)
        stats_inner.pack(fill=tk.X)

        self.up_label = ttk.Label(stats_inner, text="UP (North): 0", foreground="#16a34a", style="Count.TLabel")
        self.up_label.pack(side=tk.LEFT, expand=True)

        self.down_label = ttk.Label(stats_inner, text="DOWN (South): 0", foreground="#ea580c", style="Count.TLabel")
        self.down_label.pack(side=tk.LEFT, expand=True)

        self.tot_label = ttk.Label(stats_inner, text="TOTAL: 0", foreground="#0f172a", style="Count.TLabel")
        self.tot_label.pack(side=tk.LEFT, expand=True)

        # 5. Execution Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(4, 0))

        start_btn = ttk.Button(
            btn_frame, text="▶ Start Vehicle Counting",
            style="Action.TButton", command=self._start_processing
        )
        start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        reset_btn = ttk.Button(btn_frame, text="⟲ Reset Defaults", command=self._reset_defaults)
        reset_btn.pack(side=tk.RIGHT, padx=(6, 0))

        # Status footer
        self.status_var = tk.StringVar(value="Status: Ready to process.")
        status_lbl = ttk.Label(main_frame, textvariable=self.status_var, foreground="#64748b", font=("Helvetica", 9))
        status_lbl.pack(anchor=tk.W, pady=(10, 0))

    def _browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)

    def _reset_defaults(self):
        self.video_path_var.set(DEFAULT_VIDEO_PATH)
        self.method_var.set("MOG2")
        self.line_pos_var.set(0.55)
        self.line_lbl.config(text="55%")
        self.min_area_var.set(600)
        self.area_lbl.config(text="600")
        self.save_video_var.set(True)
        self.status_var.set("Status: Reset to defaults.")

    def _start_processing(self):
        video_path = self.video_path_var.get().strip()
        if not os.path.exists(video_path):
            messagebox.showerror("File Error", f"Video file not found:\n{video_path}")
            return

        config = SystemConfig()
        config.video_path = video_path
        config.detector.method = self.method_var.get()
        config.detector.min_area = int(self.min_area_var.get())
        config.counter.line_y_ratio = float(self.line_pos_var.get())
        config.save_output = bool(self.save_video_var.get())

        self.status_var.set("Status: Processing video in OpenCV window...")
        self.current_system = VehicleCountingSystem(config)

        # Launch in background thread so GUI remains responsive
        def runner():
            try:
                stats = self.current_system.run(
                    video_source=config.video_path,
                    headless=False,
                    save_video=config.save_output
                )
                self.root.after(0, lambda: self._update_final_stats(stats))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Execution Error", str(e)))
                self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))

        self.system_thread = threading.Thread(target=runner, daemon=True)
        self.system_thread.start()

    def _update_final_stats(self, stats: dict):
        self.up_label.config(text=f"UP (North): {stats.get('upward_count', 0)}")
        self.down_label.config(text=f"DOWN (South): {stats.get('downward_count', 0)}")
        self.tot_label.config(text=f"TOTAL: {stats.get('total_count', 0)}")
        self.status_var.set("Status: Completed video processing.")
        messagebox.showinfo(
            "Processing Complete",
            f"Finished vehicle counting.\n\n"
            f"• Upward Count   : {stats.get('upward_count', 0)}\n"
            f"• Downward Count : {stats.get('downward_count', 0)}\n"
            f"• Total Count    : {stats.get('total_count', 0)}\n\n"
            f"Output Video: {stats.get('output_video_path', 'N/A')}"
        )


def launch_gui():
    """Start Tkinter application event loop."""
    root = tk.Tk()
    app = VehicleCountingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
