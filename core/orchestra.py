from emotion import EmotionDetector


class Orchestrator:
    def __init__(self):
        self.emotion_detector = EmotionDetector()

    def process(self, text: str) -> dict:
        return {"emotion": self.emotion_detector.detect(text)}