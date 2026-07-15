from pydantic import BaseModel, Field


class EmotionScore(BaseModel):
    """A single emotion label with its confidence score."""
    label: str
    score: float = Field(ge=0.0, le=1.0)


class EmotionResult(BaseModel):
    """
    Full output of the EmotionDetector for one message.
    """
    text: str                                  # original input text
    primary_emotion: str                       # highest-confidence label (top 1)
    emotions: list[EmotionScore]               # top-k emotions, sorted desc by score
    raw_scores: dict[str, float]               # full 28-label distribution (unfiltered)