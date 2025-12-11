from ultralytics import YOLO
import cv2
import torch

from utils import (
    generate_audio_files,
    get_detection_word_and_audio,
    speak,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("model.pt").to(device)


def run_detection() -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    last_word = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        yolo_result = results[0]
        annotated = yolo_result.plot()

        arabic_word, audio_file = get_detection_word_and_audio(yolo_result, model)

        if arabic_word and audio_file and arabic_word != last_word:
            speak(audio_file)
            last_word = arabic_word

        cv2.imshow("YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_audio_files(model)
    run_detection()
