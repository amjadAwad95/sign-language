from typing import Any, Optional, Tuple

from .mappings import CLASS_EN_AR
from .audio import get_audio_file_for_english_word


def get_arabic_and_audio_for_result(
    result: Any, model: Any
) -> Tuple[Optional[str], Optional[str]]:
    """Given a single YOLO result, return (arabic_word, english_audio_file_path).

    If no mapped class is detected, both values are None.
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

    audio_file = get_audio_file_for_english_word(english_label)
    return arabic_word, audio_file
