"""
DeepFakeDetector — test GUI.

Runs the Phase 1 preprocessing pipeline (with scene-based frame selection) on a
picked video and shows the representative frames (with face boxes) plus the
aligned face crops, all from inside a small desktop window.

Run from source:   python -m gui.app
Build an .exe:     gui\\build_exe.bat
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Make `src` importable whether running from source or from a frozen exe.
if getattr(sys, "frozen", False):
    APP_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "DeepFakeDetector"
else:
    _ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_ROOT))
    APP_DIR = _ROOT

from src.preprocessing import (  # noqa: E402
    FrameExtractionConfig,
    FrameStore,
    PipelineConfig,
    process_video,
)

OUTPUT_DIR = APP_DIR / "data" / "processed"
THUMB_MAX_W = 340


# --------------------------------------------------------------------------- #
# Log capture → GUI queue
# --------------------------------------------------------------------------- #
class QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue"):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(("log", self.format(record)))
        except Exception:
            pass


class StreamToQueue:
    """Redirect stdout/stderr into the GUI queue (captures MediaPipe's logs)."""

    def __init__(self, q: "queue.Queue"):
        self.q = q

    def write(self, text: str) -> None:
        text = text.strip()
        if text:
            self.q.put(("log", text))

    def flush(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
class DeepFakeTesterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DeepFake Detector — Preprocessing Tester")
        self.root.geometry("1000x760")
        self.root.minsize(800, 600)

        self.q: "queue.Queue" = queue.Queue()
        self.worker: "threading.Thread | None" = None
        self.video_path: Path | None = None
        self._photos: list = []  # keep ImageTk references alive

        self._build_toolbar()
        self._build_preview()
        self._build_log()

        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
        logging.getLogger().addHandler(QueueLogHandler(self.q))

        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")

        self.browse_btn = ttk.Button(bar, text="Select Video...", command=self._select_video)
        self.browse_btn.grid(row=0, column=0, padx=(0, 8))

        self.video_label = ttk.Label(bar, text="No video selected", anchor="w")
        self.video_label.grid(row=0, column=1, sticky="ew")
        bar.columnconfigure(1, weight=1)

        row2 = ttk.Frame(self.root, padding=(10, 0))
        row2.pack(fill="x")

        self.use_selection = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Scene selection", variable=self.use_selection).pack(side="left")

        ttk.Label(row2, text="  Target FPS:").pack(side="left")
        self.fps_var = tk.DoubleVar(value=7.5)
        ttk.Spinbox(row2, from_=1.0, to=30.0, increment=0.5, width=5,
                    textvariable=self.fps_var).pack(side="left")

        ttk.Label(row2, text="  Model:").pack(side="left")
        self.model_text = tk.StringVar(value="1: full-range (video)")
        ttk.Combobox(row2, state="readonly", width=24, textvariable=self.model_text,
                     values=["0: short-range (selfies)", "1: full-range (video)"]).pack(side="left")

        self.process_btn = ttk.Button(row2, text="Process", command=self._start)
        self.process_btn.pack(side="right")

        self.progress = ttk.Progressbar(row2, mode="indeterminate", length=180)
        self.progress.pack(side="right", padx=8)

    def _build_preview(self) -> None:
        pane = ttk.PanedWindow(self.root, orient="vertical")
        pane.pack(fill="both", expand=True, padx=10, pady=8)

        self.notebook = ttk.Notebook(pane)
        pane.add(self.notebook, weight=3)

        self.frames_tab = self._make_tab("Representative Frames")
        self.faces_tab = self._make_tab("Aligned Faces")
        self.notebook.add(self.frames_tab, text="Representative Frames")
        self.notebook.add(self.faces_tab, text="Aligned Faces")

        self.summary_var = tk.StringVar(value="")
        ttk.Label(pane, textvariable=self.summary_var, anchor="w",
                  padding=(4, 2)).pack(fill="x")

    def _make_tab(self, _name: str) -> ttk.Frame:
        tab = ttk.Frame(self.notebook)
        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")

        tab.bind("<Enter>", lambda e: tab.bind_all("<MouseWheel>", _on_mousewheel))
        tab.bind("<Leave>", lambda e: tab.unbind_all("<MouseWheel>"))
        tab._inner = inner  # type: ignore[attr-defined]
        return tab

    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Log", padding=(6, 4))
        frame.pack(fill="both", padx=10, pady=(0, 10))
        self.log = tk.Text(frame, height=10, state="disabled", wrap="word",
                           bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")],
        )
        if path:
            self.video_path = Path(path)
            self.video_label.configure(text=str(self.video_path))

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if self.video_path is None:
            messagebox.showwarning("No video", "Select a video file first.")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._set_running(True)
        self.summary_var.set("")
        self._clear_tab(self.frames_tab)
        self._clear_tab(self.faces_tab)
        self._log("=" * 70)
        self._log(f"Processing: {self.video_path}")

        model_selection = 1 if "1:" in self.model_text.get() else 0
        config = PipelineConfig(
            frame_extraction=FrameExtractionConfig(target_fps=float(self.fps_var.get())),
            model_selection=model_selection,
            use_scene_selection=bool(self.use_selection.get()),
        )

        self.worker = threading.Thread(
            target=self._run_pipeline,
            args=(config,),
            daemon=True,
        )
        self.worker.start()

    def _run_pipeline(self, config: PipelineConfig) -> None:
        # Absorb stdout/stderr so MediaPipe C++ logs appear in the GUI log box.
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = StreamToQueue(self.q)
        sys.stderr = StreamToQueue(self.q)
        try:
            result = process_video(str(self.video_path), str(OUTPUT_DIR), config=config)
            self.q.put(("result", result))
        except Exception:
            self.q.put(("error", traceback.format_exc()))
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    # ------------------------------------------------------------------ #
    # Queue polling (main thread)
    # ------------------------------------------------------------------ #
    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "error":
                    self._set_running(False)
                    self._log(payload)
                    messagebox.showerror("Processing failed", payload)
                elif kind == "result":
                    self._set_running(False)
                    self._show_result(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ #
    # Result rendering
    # ------------------------------------------------------------------ #
    def _show_result(self, result) -> None:
        self._photos.clear()

        frames, faces = self._load_result(result)

        summary = (
            f"Frames stored: {result.num_frames_extracted}   "
            f"Faces detected: {result.num_faces_detected}   "
            f"Output: {result.output_dir}"
        )
        self.summary_var.set(summary)

        if frames:
            for i, (img, n_faces) in enumerate(frames):
                self._add_thumbnail(self.frames_tab, img,
                                    caption=f"frame {i + 1}  ({n_faces} face(s))")
        else:
            self._log("No representative frames produced.")

        if faces is not None:
            for i, face in enumerate(faces):
                self._add_thumbnail(self.faces_tab, face, caption=f"face {i + 1}")
        else:
            self._log("No aligned faces produced.")

        self._log(summary)

    def _load_result(self, result):
        """Return (frames_with_boxes, faces) as numpy arrays."""
        frames = []
        faces = None
        h5 = result.output_dir / f"{result.video_path.stem}.h5"

        if h5.exists():
            with FrameStore(h5) as store:
                frame_imgs = store.get_frames()
                bboxes = store.get_bboxes()
                frame_map = store.get_frame_map()
                for i in range(len(frame_imgs)):
                    boxes = [bboxes[j] for j in range(len(frame_map)) if frame_map[j] == i]
                    img = frame_imgs[i].copy()
                    for x, y, w, h in boxes:
                        cv2.rectangle(img, (int(x), int(y)),
                                      (int(x) + int(w), int(y) + int(h)),
                                      (0, 255, 0), 2)
                    frames.append((img, len(boxes)))
                if store.num_faces:
                    faces = store.get_faces()
        else:
            for pf in result.frames:
                if pf.frame_path and pf.frame_path.exists():
                    img = cv2.imread(str(pf.frame_path))
                    if img is None:
                        continue
                    if pf.face_box is not None:
                        b = pf.face_box
                        cv2.rectangle(img, (b.x, b.y), (b.x + b.width, b.y + b.height),
                                      (0, 255, 0), 2)
                    frames.append((img, 1 if pf.face_box else 0))
                if pf.face_path and pf.face_path.exists():
                    face = cv2.imread(str(pf.face_path))
                    if faces is None:
                        faces = []
                    faces.append(face)

        return frames, faces

    def _add_thumbnail(self, tab, image_bgr: np.ndarray, caption: str) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if pil.width > THUMB_MAX_W:
            h = int(pil.height * THUMB_MAX_W / pil.width)
            pil = pil.resize((THUMB_MAX_W, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._photos.append(photo)

        inner = tab._inner  # type: ignore[attr-defined]
        cell = ttk.Frame(inner)
        cell.pack(pady=6)
        ttk.Label(cell, image=photo).pack()
        ttk.Label(cell, text=caption, font=("Segoe UI", 9)).pack()

    def _clear_tab(self, tab) -> None:
        for child in tab._inner.winfo_children():  # type: ignore[attr-defined]
            child.destroy()

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.process_btn.configure(state=state)
        self.browse_btn.configure(state=state)
        if running:
            self.progress.start(10)
        else:
            self.progress.stop()


def _selftest(video_path: str, out_file: str) -> None:
    """Headless sanity check for the frozen exe: run the real pipeline and write
    the outcome to a file (windowed exes have no console).
        DeepFakeDetectorTester.exe --selftest <video> <outfile>
    """
    import logging as _lg

    root = _lg.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    _lg.basicConfig(
        filename=str(out_file), filemode="w", level=_lg.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
    )

    from src.preprocessing import PipelineConfig, process_video

    try:
        result = process_video(video_path, str(OUTPUT_DIR), config=PipelineConfig())
        with open(out_file, "a") as f:
            f.write(f"\nOK frames={result.num_frames_extracted} "
                    f"faces={result.num_faces_detected}\n")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        with open(out_file, "a") as f:
            f.write(f"\nERROR: {exc}\n{_tb.format_exc()}\n")
        sys.exit(1)


def main() -> None:
    if "--selftest" in sys.argv:
        if len(sys.argv) < 4:
            sys.exit("usage: DeepFakeDetectorTester.exe --selftest <video> <outfile>")
        _selftest(sys.argv[2], sys.argv[3])
    root = tk.Tk()
    DeepFakeTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
