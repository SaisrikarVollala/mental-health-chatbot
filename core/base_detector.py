from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """
    Every detector (emotion, intent, cause, severity) implements this.
    Keeps the orchestrator agnostic — it just calls .detect(text) and
    gets a structured result back, regardless of what's inside.
    """

    @abstractmethod
    def detect(self, text: str):
        """Takes raw user text, returns a Pydantic result object."""
        raise NotImplementedError

    def preprocess(self, text: str) -> str:
        """Shared default preprocessing. Override per-detector if needed."""
        return text.strip()

