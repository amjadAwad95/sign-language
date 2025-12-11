from .mappings import CLASS_EN_AR
from .audio import (
    AUDIO_DIR,
    ensure_audio_dir,
    create_audio_file,
    generate_audio_files,
    get_audio_file,
    speak,
)
from .detection import get_detection_word_and_audio

__all__ = [
    "CLASS_EN_AR",
    "AUDIO_DIR",
    "ensure_audio_dir",
    "create_audio_file",
    "generate_audio_files",
    "get_audio_file",
    "speak",
    "get_detection_word_and_audio",
]
