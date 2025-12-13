import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

from utils import get_detection_word_and_audio


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = YOLO("model.pt").to(device)

last_word: str | None = None


def detection(frame: np.ndarray, conf_threshold: float):
    global last_word

    if frame is None:
        return None, "", None

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = model(frame_bgr, conf=conf_threshold, verbose=False, imgsz=416)
    result = results[0]

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    arabic_word, audio_file = get_detection_word_and_audio(result, model)
    next_audio = None
    if arabic_word and arabic_word != last_word:
        last_word = arabic_word
        next_audio = audio_file

    return annotated_rgb, arabic_word or last_word or "", next_audio


def launch_app() -> None:
    with gr.Blocks(title="Sign Language Live") as demo:
        gr.Markdown("# Live Sign Language Detection\nWebcam streaming with on-frame annotations.")

        with gr.Row():
            cam = gr.Image(
                sources="webcam",
                streaming=True,
                type="numpy",
                label="Camera",
                height=240,
                width=320,
            )
            annotated = gr.Image(
                type="numpy",
                label="Detections",
                height=240,
                width=320,
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
            fn=detection,
            inputs=[cam, conf_threshold],
            outputs=[annotated, word, audio],
        )

    demo.launch(share=True)


if __name__ == "__main__":
    launch_app()
