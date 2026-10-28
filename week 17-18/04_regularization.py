"""
04_regularization.py
- Deep Learning Regularization:
  1. Dropout
  2. Batch Normalization
  3. Early Stopping
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def check(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class RegularizedNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=2):
        super(RegularizedNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 1. Batch Normalization
            nn.ReLU(),
            nn.Dropout(p=0.3),          # 2. Dropout
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def run_regularization():
    print("=" * 60)
    print("[Step 4] Regularization: Dropout, BatchNorm & Early Stopping")
    print("=" * 60)

    torch.manual_seed(42)
    X_train, y_train = torch.randn(100, 20), torch.randint(0, 2, (100,))
    X_val, y_val = torch.randn(30, 20), torch.randint(0, 2, (30,))

    model = RegularizedNet(input_dim=20, hidden_dim=64, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    early_stopping = EarlyStopping(patience=3)

    print("Training with Early Stopping monitoring validation loss (Patience=3)...")

    for epoch in range(1, 30):
        # Training Phase (Dropout active)
        model.train()
        optimizer.zero_grad()
        train_out = model(X_train)
        train_loss = criterion(train_out, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation Phase (Dropout inactive)
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()

        print(f"Epoch [{epoch:02d}] - Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss:.4f}")

        if early_stopping.check(val_loss):
            print(f"[STOP] Early stopping triggered at Epoch {epoch}! (Validation loss stopped improving)")
            break

    print("[SUCCESS] Regularization and Early Stopping demo completed!\n")

if __name__ == "__main__":
    run_regularization()

