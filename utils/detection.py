from typing import Any, Optional, Tuple

from .mappings import CLASS_EN_AR
from .audio import get_audio_file


def get_detection_word_and_audio(
    result: Any, model: Any
) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a single YOLO result, return (arabic_word, english_audio_file_path).
    If no mapped class is detected, both values are None.
    :param result: The YOLO result object.
    :param model: The YOLO model with class names.
    :return: A tuple of (arabic_word, audio_file_path) or (None, None).
    """

    if result is None or getattr(result, "boxes", None) is None:
        return None, None

    boxes = result.boxes
    if getattr(boxes, "cls", None) is None or len(boxes.cls) == 0:
        return None, None

    class_id = int(boxes.cls[0].item())
    english_label = model.names.get(class_id)
    if not english_label:
        return None, None

    arabic_word = CLASS_EN_AR.get(english_label)
    if not arabic_word:
        return None, None

    audio_file = get_audio_file(english_label)
    return arabic_word, audio_file
