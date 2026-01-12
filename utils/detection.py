from typing import Any, Optional

def get_detection_word(result: Any, model: Any) -> Optional[str]:
    """
    Given a single YOLO result, return english_label.
    If no mapped class is detected, the value is None.
    :param result: The YOLO result object.
    :param model: The YOLO model with class names.
    :return: The English label corresponding to the detected class or None if not found.
    """

    if result is None or getattr(result, "boxes", None) is None:
        return None
    
    boxes = result.boxes
    if getattr(boxes, "cls", None) is None or len(boxes.cls) == 0:
        return None

    class_id = int(boxes.cls[0].item())
    english_label = model.names.get(class_id)
    if not english_label:
        return None
    
    return english_label
