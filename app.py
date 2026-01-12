import cv2
import torch
from ultralytics import YOLO
from speakers import PygameSpeaker
from utils import get_detection_word

MODEL_PATH = "model/model.onnx"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)

pygame_speaker = PygameSpeaker()
pygame_speaker.initialize(model=model)


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

        english_word = get_detection_word(yolo_result, model)

        if english_word and english_word != last_word:
            pygame_speaker.speak(english_word)
            last_word = english_word
        cv2.imshow("YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
