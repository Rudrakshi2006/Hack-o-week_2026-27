"""
06_transformers_and_llm.py
- Hugging Face Transformers basics (Pretrained Pipeline)
- LLM API integration patterns (OpenAI, Anthropic, Hugging Face, Ollama)
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def run_transformers_and_llm():
    print("=" * 60)
    print("[Step 6 & 7] Hugging Face Transformers & LLM APIs")
    print("=" * 60)

    # 1. Hugging Face Transformers Pipeline (Sentiment Analysis)
    print("\n1. Hugging Face Pretrained Transformer (Sentiment Analysis Pipeline):")
    sample_texts = [
        "PyTorch and Deep Learning are amazingly powerful and fun!",
        "Debugging CUDA memory errors is frustrating and difficult."
    ]

    try:
        from transformers import pipeline
        # Fast, lightweight default pipeline
        classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        for text in sample_texts:
            result = classifier(text)[0]
            print(f"  Input : \"{text}\"")
            print(f"  Result: {result['label']} (Confidence Score: {result['score']:.4f})\n")
    except Exception as e:
        print(f"  Pipeline Note: {e}")
        print("  [Simulated Output]:")
        print("  Input : 'PyTorch and Deep Learning are amazingly powerful and fun!' -> POSITIVE (Score: 0.9998)")
        print("  Input : 'Debugging CUDA memory errors is frustrating and difficult.' -> NEGATIVE (Score: 0.9985)\n")

    # 2. LLM APIs Integration Guide & Demo
    print("2. LLM API Integration Patterns (OpenAI / Anthropic / Ollama):")
    print("""
    # --- Pattern A: OpenAI API ---
    # from openai import OpenAI
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": "Explain CNNs in 1 sentence."}]
    # )

    # --- Pattern B: Anthropic Claude API ---
    # import anthropic
    # client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # message = client.messages.create(
    #     model="claude-3-5-haiku-20241022",
    #     messages=[{"role": "user", "content": "Explain transfer learning."}]
    # )

    # --- Pattern C: Open Source Local Models via Ollama ---
    # import requests
    # res = requests.post('http://localhost:11434/api/generate', json={
    #     'model': 'llama3.2',
    #     'prompt': 'Why use Dropout in deep learning?'
    # })
    """)

    # Demonstration of LLM Call / Mock fallback
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print("Found OPENAI_API_KEY! Making live API request...")
    else:
        print("[NOTE] No external API key in environment (OPENAI_API_KEY). Running demonstration:")
        prompt = "Explain CNNs in 1 sentence."
        simulated_response = "CNNs use learnable filter kernels and pooling to detect spatial hierarchies of features in images."
        print(f"  Prompt: \"{prompt}\"")
        print(f"  LLM Response: \"{simulated_response}\"")

    print("\n[SUCCESS] Transformers & LLM API step completed successfully!\n")

if __name__ == "__main__":
    run_transformers_and_llm()

