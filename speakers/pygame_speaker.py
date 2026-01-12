import re
from pathlib import Path
import pygame
from gtts import gTTS
from mappings import CLASS_EN_AR
from .bace_speaker import BaseSpeaker


def _safe_filename(name: str) -> str:
    """
    Make a safe filename from a label:
    - remove invalid characters
    - collapse spaces
    :param name: The original label/name.
    :return: A safe filename string.
    """
    name = name.strip()
    name = re.sub(r"[^\w\s\-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name or "unknown"


class PygameSpeaker(BaseSpeaker):
    """
    Speaker implementation using pygame and gTTS for Arabic text-to-speech.
    """

    def __init__(self, audio_dir: str = "audio") -> None:
        """
        Initialize the PygameSpeaker.
        :param audio_dir: Directory to store audio files.
        """
        self.audio_dir = audio_dir

        self.audio_path = (Path(__file__).resolve().parent / ".." / audio_dir).resolve()
        self._mixer_initialized = False
        self._label_to_file: dict[str, Path] = {}

    def _init_mixer_once(self) -> None:
        """
        Initialize the pygame mixer if not already initialized.
        """
        if self._mixer_initialized:
            return
        try:
            pygame.mixer.init()
            self._mixer_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pygame mixer: {e}") from e

    def initialize(self, model) -> None:
        """
        Generate missing audio files for each class in the model.
        :param model: The object detection model with class names.
        """
        print("Initializing PygameSpeaker...")
        self.audio_path.mkdir(parents=True, exist_ok=True)
        print(f"Audio directory: {self.audio_path}")

        for _, english_label in model.names.items():
            safe_label = _safe_filename(str(english_label))
            file_path = self.audio_path / f"{safe_label}.mp3"
            self._label_to_file[str(english_label)] = file_path

        print("Generating missing audio files (if any)...")
        generated = 0

        for english_label, file_path in self._label_to_file.items():
            if file_path.exists():
                continue

            arabic_text = CLASS_EN_AR.get(english_label, english_label)

            try:
                tts = gTTS(text=arabic_text, lang="ar")
                tts.save(str(file_path))
                generated += 1
            except Exception as e:
                print(
                    f"Failed to generate audio for '{english_label}' -> {file_path.name}: {e}"
                )

        print(f"Audio generation done. Created {generated} new file(s).")

    def speak(self, message: str) -> None:
        """
        Speak the given message (English label).
        :param message: The message/label to speak.
        """
        self._init_mixer_once()

        audio_file = self._label_to_file.get(message)
        if audio_file is None:
            safe_label = _safe_filename(message)
            audio_file = self.audio_path / f"{safe_label}.mp3"

        if not audio_file.exists():
            print(f"Audio file not found for '{message}': {audio_file}")
            return

        try:
            pygame.mixer.music.load(str(audio_file))
            pygame.mixer.music.play()

        except Exception as e:
            print(f"Error playing '{audio_file}': {e}")
