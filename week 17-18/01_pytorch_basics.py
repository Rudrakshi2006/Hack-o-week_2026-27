"""
01_pytorch_basics.py
- Tensors creation & operations
- Autograd (Automatic differentiation)
- Basic Training Loop (Linear Regression)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim

def run_basics():
    print("=" * 60)
    print("[Step 1] PyTorch Basics: Tensors, Autograd & Training Loop")
    print("=" * 60)

    # 1. Tensors
    print("\n1. Tensor Creation:")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"Tensor x:\n{x}\nShape: {x.shape}, Dtype: {x.dtype}")

    # 2. Autograd
    print("\n2. Autograd (Automatic Differentiation):")
    w = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    y = w * 3.0 + b  # y = 3w + b
    y.backward()
    print(f"Equation: y = 3*w + b")
    print(f"dy/dw = {w.grad.item()} (Expected: 3.0)")
    print(f"dy/db = {b.grad.item()} (Expected: 1.0)")

    # 3. Simple Training Loop (Fit y = 2x + 1)
    print("\n3. Basic Training Loop (Simple Linear Model):")
    X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # y = 2x + 1

    model = nn.Linear(in_features=1, out_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(1, 101):
        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(f"Epoch [{epoch}/100] - Loss: {loss.item():.4f}")

    # Inference test
    test_x = torch.tensor([[5.0]])
    pred_y = model(test_x).item()
    print(f"\nInference: For x = 5.0, Predicted y = {pred_y:.2f} (Expected: ~11.0)")
    print("[SUCCESS] PyTorch basics completed successfully!\n")

if __name__ == "__main__":
    run_basics()

