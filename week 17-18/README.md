# 🚀 Deep Learning Crash Course Project

A simple, modular, and non-complex Deep Learning starter project covering all fundamental topics in PyTorch, Computer Vision, Sequential NLP, Transformers, and LLM APIs.

---

## 📁 Project Structure

| File | Topic Covered | Details |
|------|---------------|---------|
| [01_pytorch_basics.py](file:///c:/week%2017-18/01_pytorch_basics.py) | **PyTorch Basics** | Tensors, Autograd (`backward()`), and a basic 1D Linear Regression training loop. |
| [02_cnn_classification.py](file:///c:/week%2017-18/02_cnn_classification.py) | **Convolutional Neural Networks (CNN)** | Convolutions (`Conv2d`), Max Pooling (`MaxPool2d`), Flatten & Linear layers for image classification. |
| [03_rnn_sequential.py](file:///c:/week%2017-18/03_rnn_sequential.py) | **Sequence Modeling (LSTM/RNN)** | Handling sequential data with PyTorch's `nn.LSTM` for sequence classification. |
| [04_regularization.py](file:///c:/week%2017-18/04_regularization.py) | **Regularization Techniques** | Dropout (`nn.Dropout`), Batch Normalization (`nn.BatchNorm1d`), and custom **Early Stopping**. |
| [05_transfer_learning.py](file:///c:/week%2017-18/05_transfer_learning.py) | **Transfer Learning** | Loading pretrained MobileNetV2, freezing the backbone feature extractor, and fine-tuning the classification head. |
| [06_transformers_and_llm.py](file:///c:/week%2017-18/06_transformers_and_llm.py) | **Hugging Face & LLM APIs** | Pretrained Transformer sentiment analysis pipeline + OpenAI, Claude, and Ollama API integration patterns. |
| [run_all.py](file:///c:/week%2017-18/run_all.py) | **Master Runner** | Executes all 6 modules sequentially with clean logs and timing. |

---

## ⚡ How to Run

### Run Everything Together (All Modules in ~1 Minute)
```bash
python run_all.py
```

### Run Any Specific Module Individually
```bash
python 01_pytorch_basics.py
python 02_cnn_classification.py
python 03_rnn_sequential.py
python 04_regularization.py
python 05_transfer_learning.py
python 06_transformers_and_llm.py
```

