from ultralytics import YOLO
import cv2
import torch

from utils import (
    generate_all_class_audio_files,
    get_arabic_and_audio_for_result,
    play_audio_file,
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

        results = model(frame)
        yolo_result = results[0]
        annotated = yolo_result.plot()

        arabic_word, audio_file = get_arabic_and_audio_for_result(yolo_result, model)

        if arabic_word:
            cv2.putText(
                annotated,
                arabic_word,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if arabic_word and audio_file and arabic_word != last_word:
            print(f"Detected word: {arabic_word} | Audio file: {audio_file}")
            play_audio_file(audio_file)
            last_word = arabic_word

        cv2.imshow("YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Generate the Arabic audio files once (idempotent) and then run detection
    generate_all_class_audio_files(model)
    run_detection()
