import spacy
from transformers import pipeline
import warnings

# Suppress some noisy warnings from transformers if any
warnings.filterwarnings("ignore")

# Load models once so they can be reused
print("Loading models... This may take a moment.")

# Emotion
try:
    from emotion.service import EmotionService
    emotion_service = EmotionService()
except ImportError:
    print("Warning: Could not import EmotionService. Make sure you run this script from the 'mental-health-chatbot' directory.")
    emotion_service = None

# Cause
cause_pipeline = pipeline(
    "token-classification",
    model="tanfiona/unicausal-tok-baseline",
    aggregation_strategy="simple"
)

# Intent
nlp_intent = spacy.load("en_core_web_trf")

def extract_intent(text):
    doc = nlp_intent(text)
    for token in doc:
        if token.dep_ == "dobj":
            action = token.head.lemma_
            obj = token.text
            return action, obj
    return None, None

# Severity
severity_classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=-1
)

candidate_labels = [
    "mild emotional distress that does not require immediate professional help",
    "moderate mental health concern that should be evaluated by a mental health professional",
    "mental health crisis requiring immediate medical or emergency intervention"
]

label_mapping = {
    candidate_labels[0]: "Low",
    candidate_labels[1]: "Medium",
    candidate_labels[2]: "High"
}

def analyze_query(query: str):
    print(f"\nAnalyzing query: '{query}'\n" + "-"*50)
    
    # 1. Emotion
    if emotion_service:
        primary_emotion = emotion_service.get_primary_emotion(query)
        print(f"Emotion : {primary_emotion}")
    else:
        primary_emotion = None
        print("Emotion : Not available")
    
    # 2. Cause
    cause_result = cause_pipeline(query)
    # Extract just the words of the cause if present
    causes = [entity['word'] for entity in cause_result] if cause_result else []
    print(f"Cause   : {', '.join(causes) if causes else 'None detected'}")
    
    # 3. Intent
    action, obj = extract_intent(query)
    if action and obj:
        print(f"Intent  : Action='{action}', Object='{obj}'")
    else:
        print("Intent  : None detected")
    
    # 4. Severity
    severity_result = severity_classifier(
        query,
        candidate_labels=candidate_labels,
        hypothesis_template="This statement describes {}.",
        multi_label=False
    )
    predicted_severity = label_mapping[severity_result["labels"][0]]
    print(f"Severity: {predicted_severity} (Confidence: {severity_result['scores'][0]:.4f})")
    print("-" * 50)
    
    return {
        "emotion": primary_emotion,
        "cause": causes,
        "intent": {"action": action, "object": obj},
        "severity": predicted_severity
    }

if __name__ == "__main__":
    print("\nPipeline Ready!")
    while True:
        try:
            user_input = input("\nEnter a query (or type 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input.strip():
                analyze_query(user_input)
        except (KeyboardInterrupt, EOFError):
            break
