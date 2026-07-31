import ollama

class AbstractiveSummarizer:
    def __init__(self, model_name="gemma3:4b"):
        self.model_name = model_name
        print(f"\nInitializing Summarizer using Ollama (Model: {self.model_name})...")
        
        # We assume the model is already pulled via `ollama pull gemma3:4b`
        try:
            # Quick check if model is available
            ollama.show(self.model_name)
            print("Summarizer loaded successfully!")
            self.loaded = True
        except Exception as e:
            print(f"Warning: Model {self.model_name} not found in Ollama. Please run `ollama pull {self.model_name}` in your terminal.")
            self.loaded = False

    def generate_response(self, combined_context: dict) -> str:
        if not self.loaded:
            return "Summarizer model is not loaded. Please make sure Ollama is running and the model is pulled."

        system_prompt = (
            "You are a compassionate, empathetic mental health assistant. "
            "You are given a user query, a dialog analysis (emotions, severity, intent), "
            "and relevant background knowledge from psychology books. "
            "Write a supportive and helpful response to the user. "
            "Ground your advice in the provided knowledge contexts but keep your tone conversational and empathetic. "
            "Do not just summarize the texts; offer practical, compassionate advice."
        )
        
        user_message = f"User Query: {combined_context.get('query', '')}\n\n"
        
        user_message += "--- Dialog Analysis ---\n"
        da = combined_context.get("dialog_analysis", {})
        user_message += f"Emotion: {da.get('emotion', 'None')}\n"
        user_message += f"Severity: {da.get('severity', 'None')}\n"
        intent = da.get("intent", {})
        if intent.get("action"):
            user_message += f"Intent: Action='{intent['action']}', Object='{intent['object']}'\n"
            
        user_message += "\n--- Relevant Knowledge ---\n"
        for i, ctx in enumerate(combined_context.get("knowledge_contexts", [])):
            user_message += f"[{i+1}] Source: {ctx['source']}\nText: {ctx['text'][:500]}...\n\n"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        print("Generating response...")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error generating response from Ollama: {e}"
