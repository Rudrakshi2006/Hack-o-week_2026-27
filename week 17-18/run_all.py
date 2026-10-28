"""
run_all.py
Master runner script that executes all deep learning modules sequentially:
1. PyTorch Basics (Tensors, Autograd, Training Loop)
2. CNNs (Convolutions, Pooling, Classification)
3. RNNs / LSTMs (Sequence Modeling)
4. Regularization (Dropout, BatchNorm, Early Stopping)
5. Transfer Learning (Pretrained CNNs)
6. Transformers & LLM APIs (Hugging Face & LLM integration)
"""

import sys
import time
import importlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

modules = [
    ("01_pytorch_basics", "run_basics"),
    ("02_cnn_classification", "run_cnn"),
    ("03_rnn_sequential", "run_rnn"),
    ("04_regularization", "run_regularization"),
    ("05_transfer_learning", "run_transfer_learning"),
    ("06_transformers_and_llm", "run_transformers_and_llm"),
]

def main():
    print("=" * 70)
    print("DEEP LEARNING CRASH COURSE - ALL-IN-ONE PROJECT")
    print("Topics: PyTorch, CNN, RNN/LSTM, Regularization, Transfer Learning, HF & LLM")
    print("=" * 70)
    
    start_total = time.time()
    
    for mod_name, func_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
            t0 = time.time()
            func()
            t_taken = time.time() - t0
            print(f"[TIME] Finished {mod_name} in {t_taken:.2f}s\n")
        except Exception as e:
            print(f"[ERROR] Error in {mod_name}: {e}\n")

    total_time = time.time() - start_total
    print("=" * 70)
    print(f"[DONE] ALL MODULES EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 70)

if __name__ == "__main__":
    main()

