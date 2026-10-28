"""
03_rnn_sequential.py
- Sequence Modeling basics with LSTM / GRU / RNN
- Sequential data processing
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, num_layers=1, num_classes=2):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out)

def run_rnn():
    print("=" * 60)
    print("[Step 3] Sequence Modeling: RNN / LSTM / GRU Basics")
    print("=" * 60)

    torch.manual_seed(42)
    seq_data = torch.randn(20, 10, 5)
    labels = torch.randint(0, 2, (20,))

    model = SimpleLSTMModel(input_dim=5, hidden_dim=16, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"Sequence Input Shape: {seq_data.shape} (Batch=20, SeqLen=10, Features=5)")

    for epoch in range(1, 6):
        optimizer.zero_grad()
        outputs = model(seq_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean() * 100
        print(f"Epoch [{epoch}/5] - Loss: {loss.item():.4f} - Accuracy: {acc:.1f}%")

    print("[SUCCESS] Sequence modeling (LSTM) step completed successfully!\n")

if __name__ == "__main__":
    run_rnn()

