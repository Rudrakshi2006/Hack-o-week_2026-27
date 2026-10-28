"""
02_cnn_classification.py
- Convolutional Neural Network (CNN)
- Convolutions, Max Pooling, Image Classification
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Conv block: Conv2D -> ReLU -> MaxPool2D
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

def run_cnn():
    print("=" * 60)
    print("[Step 2] CNN: Convolutions, Pooling & Classification")
    print("=" * 60)

    torch.manual_seed(42)
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 2, (16,))

    model = SimpleCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print(f"Input Shape: {images.shape} (Batch=16, Channels=3, H=32, W=32)")

    # Training for 5 epochs
    for epoch in range(1, 6):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean() * 100
        print(f"Epoch [{epoch}/5] - Loss: {loss.item():.4f} - Accuracy: {acc:.1f}%")

    print("[SUCCESS] CNN image classification step completed successfully!\n")

if __name__ == "__main__":
    run_cnn()

