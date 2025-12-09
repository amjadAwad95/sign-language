from .mappings import CLASS_EN_AR
from .audio import (
    AUDIO_DIR,
    ensure_audio_dir,
    create_english_audio_file,
    generate_all_class_audio_files,
    get_audio_file_for_english_word,
    play_audio_file,
)
from .detection import get_arabic_and_audio_for_result

__all__ = [
    "CLASS_EN_AR",
    "AUDIO_DIR",
    "ensure_audio_dir",
    "create_english_audio_file",
    "generate_all_class_audio_files",
    "get_audio_file_for_english_word",
    "play_audio_file",
    "get_arabic_and_audio_for_result",
]
