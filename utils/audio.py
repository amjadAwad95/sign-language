import os
import pygame
from gtts import gTTS
from typing import Any, Optional
from .mappings import CLASS_EN_AR


AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "audio")
AUDIO_DIR = os.path.abspath(AUDIO_DIR)
_MIXER_INITIALIZED = False


def ensure_audio_dir() -> None:
    """
    Ensure the audio directory exists.
    """
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR, exist_ok=True)


def create_audio_file(english_word: str) -> str:
    """
    Create (or reuse) an MP3 file that speaks the mapped Arabic word.
    The file is still named with the English word (e.g. "Hello.mp3"),
    but the spoken audio is in Arabic using CLASS_EN_AR.
    :param english_word: The English word to create an audio file for.
    :return: The file path of the created or existing audio file.
    """

    ensure_audio_dir()
    file_path = os.path.join(AUDIO_DIR, f"{english_word}.mp3")

    if not os.path.exists(file_path):
        arabic_text = CLASS_EN_AR.get(english_word, english_word)
        tts = gTTS(text=arabic_text, lang="ar")
        tts.save(file_path)

    return file_path


def generate_audio_files(model: Any) -> None:
    """
    Generate Arabic audio files once for all YOLO classes.
    Files are stored with English filenames but contain Arabic speech.
    :param model: The YOLO model with class names.
    """

    for _, english_label in model.names.items():
        create_audio_file(english_label)


def get_audio_file(english_word: str) -> Optional[str]:
    """
    Return the audio file path for a given English word, if it exists.
    :param english_word: The English word to get the audio file for.
    :return: The file path if it exists, else None.
    """

    file_path = os.path.join(AUDIO_DIR, f"{english_word}.mp3")
    return file_path if os.path.exists(file_path) else None


def _ensure_mixer_initialized() -> None:
    """
    Ensure the pygame mixer is initialized for audio playback.
    """
    global _MIXER_INITIALIZED
    if not _MIXER_INITIALIZED:
        try:
            pygame.mixer.init()
            _MIXER_INITIALIZED = True
        except Exception:
            _MIXER_INITIALIZED = False


def speak(file_path: str) -> None:
    """
    Play an audio file path if it exists (non-blocking for the main loop).
    :param file_path: The path to the audio file to play.
    """

    if not file_path or not os.path.exists(file_path):
        return

    _ensure_mixer_initialized()
    if not _MIXER_INITIALIZED:
        return

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception:
        pass
