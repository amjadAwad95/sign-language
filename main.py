import os
import time
import cv2
import gradio as gr
import numpy as np
import torch
import threading
import tempfile
import shutil
from collections import deque
from typing import Optional

from ultralytics import YOLO
from utils import get_detection_word_and_audio

# =============================
# CONFIG
# =============================
MODEL_PATH = "model.onnx"

# Keep camera + inference same size to avoid extra resizing overhead
UI_SIZE = 320  # Camera displayed size
INFER_SIZE = 320  # Inference size (try 256 if CPU is slow)

INFER_EVERY_N_FRAMES = 2  # 1 = every frame, 2 or 3 = faster on CPU
AUDIO_COOLDOWN_SEC = 1.2
AUDIO_QUEUE_MAX = 5

# =============================
# MODEL
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO(MODEL_PATH)

# =============================
# SHARED STATE
# =============================
lock = threading.Lock()
stop_event = threading.Event()
worker_thread: Optional[threading.Thread] = None

latest_frame_rgb: Optional[np.ndarray] = None
latest_annotated_rgb: Optional[np.ndarray] = None
latest_word: str = ""
latest_conf: float = 0.30

audio_queue = deque(maxlen=AUDIO_QUEUE_MAX)

_last_word: Optional[str] = None
_last_audio_time: float = 0.0
_frame_counter: int = 0

TMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), "gradio_sign_audio")
os.makedirs(TMP_AUDIO_DIR, exist_ok=True)


# =============================
# HELPERS
# =============================
def draw_boxes_fast(bgr: np.ndarray, result) -> np.ndarray:
    out = bgr.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return out

    try:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except Exception:
        return out

    return out


def wait_until_file_stable(
    path: str, timeout: float = 1.0, interval: float = 0.05
) -> bool:
    """
    Some audio generators write the file gradually.
    If we send it too early, browser may play only part of it.
    """
    if not path or not os.path.exists(path):
        return False

    start = time.time()
    last_size = -1
    stable_count = 0

    while time.time() - start < timeout:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1

        if size > 0 and size == last_size:
            stable_count += 1
            if stable_count >= 2:  # stable for 2 checks
                return True
        else:
            stable_count = 0

        last_size = size
        time.sleep(interval)

    return os.path.getsize(path) > 0


def cache_bust_audio(audio_path: str) -> Optional[str]:
    """
    Make a unique copy so the browser doesn't cache and skip replay.
    Also ensures the file is fully written.
    """
    if not audio_path or not os.path.exists(audio_path):
        return None

    if not wait_until_file_stable(audio_path, timeout=1.2):
        return None

    base = os.path.basename(audio_path)
    name, ext = os.path.splitext(base)
    unique = f"{name}_{int(time.time() * 1000)}{ext}"
    new_path = os.path.join(TMP_AUDIO_DIR, unique)

    # copy to unique path
    shutil.copy2(audio_path, new_path)

    # final sanity check
    if os.path.exists(new_path) and os.path.getsize(new_path) > 0:
        return new_path
    return None


# =============================
# WORKER THREAD (YOLO)
# =============================
def inference_loop():
    global latest_frame_rgb, latest_annotated_rgb, latest_word
    global _last_word, _last_audio_time, _frame_counter

    while not stop_event.is_set():
        with lock:
            frame = None if latest_frame_rgb is None else latest_frame_rgb
            conf = float(latest_conf)

        if frame is None:
            time.sleep(0.01)
            continue

        _frame_counter += 1
        if INFER_EVERY_N_FRAMES > 1 and (_frame_counter % INFER_EVERY_N_FRAMES != 0):
            time.sleep(0.001)
            continue

        # Keep size consistent
        try:
            frame_small = cv2.resize(
                frame, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_AREA
            )
            frame_bgr = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)
        except Exception:
            time.sleep(0.01)
            continue

        # YOLO inference
        try:
            results = model(frame_bgr, conf=conf, verbose=False, device=device)
            result = results[0]
        except Exception:
            with lock:
                latest_annotated_rgb = frame_small
            time.sleep(0.01)
            continue

        annotated_bgr = draw_boxes_fast(frame_bgr, result)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        arabic_word, audio_file = get_detection_word_and_audio(result, model)

        now = time.time()
        new_audio: Optional[str] = None
        display_word = _last_word or ""

        if arabic_word:
            display_word = arabic_word

            # cooldown + change gate
            if (
                arabic_word != _last_word
                and (now - _last_audio_time) >= AUDIO_COOLDOWN_SEC
            ):
                _last_word = arabic_word
                _last_audio_time = now

                if audio_file:
                    new_audio = cache_bust_audio(audio_file)

        with lock:
            latest_annotated_rgb = annotated_rgb
            latest_word = display_word
            if new_audio:
                audio_queue.append(new_audio)

        time.sleep(0.001)


# =============================
# GRADIO STREAM CALLBACK (FAST)
# =============================
def on_stream(frame: np.ndarray, conf_threshold: float):
    global latest_frame_rgb, latest_conf

    if frame is None:
        return None, "", None

    # Force consistent UI size (reduces lag / re-layout in browser)
    frame = cv2.resize(frame, (UI_SIZE, UI_SIZE), interpolation=cv2.INTER_AREA)

    with lock:
        latest_frame_rgb = frame
        latest_conf = float(conf_threshold)

        out_img = latest_annotated_rgb if latest_annotated_rgb is not None else frame
        out_word = latest_word or ""
        out_audio = audio_queue.popleft() if len(audio_queue) > 0 else None

    return out_img, out_word, out_audio


# =============================
# APP
# =============================
def launch_app():
    global worker_thread

    if worker_thread is None or not worker_thread.is_alive():
        stop_event.clear()
        worker_thread = threading.Thread(target=inference_loop, daemon=True)
        worker_thread.start()

    with gr.Blocks(title="Sign Language Live (Stable)") as demo:
        gr.Markdown(
            "# Live Sign Language Detection (Stable Audio + Faster UI)\n"
            "**Tip:** If audio doesn’t play, click once on the page (browser autoplay policy)."
        )

        with gr.Row():
            cam = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Camera",
                height=UI_SIZE,
                width=UI_SIZE,
            )
            annotated = gr.Image(
                type="numpy",
                label="Detections",
                height=UI_SIZE,
                width=UI_SIZE,
            )

        with gr.Row():
            word = gr.Textbox(label="Detected Word", interactive=False)
            audio = gr.Audio(label="Spoken Arabic", autoplay=True, type="filepath")
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
            )

        cam.stream(
            fn=on_stream,
            inputs=[cam, conf_threshold],
            outputs=[annotated, word, audio],
        )

    demo.launch(share=True)


def shutdown():
    stop_event.set()
    if worker_thread and worker_thread.is_alive():
        worker_thread.join(timeout=1.0)


if __name__ == "__main__":
    try:
        launch_app()
    finally:
        shutdown()
