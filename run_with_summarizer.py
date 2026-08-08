import warnings
import json
import os
from pipeline import analyze_query
from knowledge.retrieve import HybridRetriever
from summarizer import AbstractiveSummarizer

warnings.filterwarnings("ignore")

def main():
    print("\n" + "=" * 60)
    print("🧠 Mental Health Chatbot — End-to-End Pipeline (with Summarizer)")
    print("=" * 60)
    
    print("\nLoading Knowledge Retriever...")
    try:
        retriever = HybridRetriever()
    except Exception as e:
        print(f"Warning: Could not load Knowledge Retriever: {e}")
        retriever = None

    print("\nLoading Abstractive Summarizer...")
    try:
        summarizer = AbstractiveSummarizer()
    except Exception as e:
        print(f"Warning: Could not load Summarizer: {e}")
        summarizer = None
        
    print("\n🤖 Ready! (Type 'quit' or 'exit' to stop)")
    print("-" * 60)
    
    while True:
        try:
            print("\n👤 You (paste text, press Enter twice to submit): ")
            lines = []
            while True:
                try:
                    line = input()
                    if not line.strip() and lines:  # Stop on empty line if we have text
                        break
                    elif not line.strip() and not lines: # Ignore leading empty lines
                        continue
                    
                    if line.lower() in ['quit', 'exit', 'q']:
                        print("\nGoodbye!")
                        return
                        
                    lines.append(line)
                except EOFError:
                    break
            
            user_input = "\n".join(lines)
            
            if user_input.strip():
                # 1. Run Dialog Analysis (from pipeline.py)
                print("\n--- Running Dialog Analysis ---")
                dialog_results = analyze_query(user_input)
                
                # 2. Run Knowledge Retrieval
                knowledge_results = []
                if retriever:
                    print("\n--- Retrieving Knowledge ---")
                    retrieved_docs = retriever.retrieve_top_k(user_input, k=5, similarity_pool=10)
                    knowledge_results = [
                        {
                            "text": r["text"],
                            "source": r["source"],
                            "relevance": round(r["relevance"], 4),
                            "score": round(r["score"], 4)
                        }
                        for r in retrieved_docs
                    ]
                
                # 3. Print Retrieved Knowledge
                if knowledge_results:
                    print("\n" + "-" * 60)
                    print("📚 Top 5 Retrieved Paragraphs:")
                    print("-" * 60)
                    for i, r in enumerate(knowledge_results, 1):
                        print(f"[{i}] Source: {r['source']} | Relevance: {r['relevance']} | MMR: {r['score']}")
                        print(f"    {r['text']}")
                        print("-" * 60)
                
                # 4. Combine contexts
                combined_context = {
                    "query": user_input,
                    "dialog_analysis": dialog_results,
                    "knowledge_contexts": knowledge_results
                }
                
                # Save to JSON for debugging
                os.makedirs("data", exist_ok=True)
                with open("data/latest_context.json", "w") as f:
                    json.dump(combined_context, f, indent=4)
                
                # 5. Generate Final Response
                if summarizer:
                    print("\n" + "─" * 60)
                    print("📝 Generating Abstractive Response (Gemma)...")
                    
                    response = summarizer.generate_response(combined_context)
                    
                    print("\n" + "═" * 60)
                    print("💡 BOT RESPONSE")
                    print("═" * 60)
                    print(response)
                    print("═" * 60)
                else:
                    print("\n[Summarizer not loaded. Skipping response generation]")
                
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
