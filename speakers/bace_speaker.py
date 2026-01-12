from abc import ABC, abstractmethod

class BaseSpeaker(ABC):
    """
    Abstract base class for speaker implementations.
    Defines the interface for speaking messages.
    """
    @abstractmethod
    def speak(self, message: str) -> None:
        """
        Speak the given message.
        :param message: The message to be spoken.
        """
        pass