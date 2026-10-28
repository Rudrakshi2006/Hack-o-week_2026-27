"""
05_transfer_learning.py
- Transfer Learning with Pretrained CNN (MobileNetV2 / ResNet)
- Feature extraction: Freeze backbone, replace final classification head
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

def run_transfer_learning():
    print("=" * 60)
    print("[Step 5] Transfer Learning with Pretrained CNN")
    print("=" * 60)

    print("Loading pretrained MobileNetV2 architecture...")
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        print("Successfully loaded MobileNetV2 with ImageNet pretrained weights.")
    except Exception as e:
        print(f"Loading MobileNetV2 architecture (offline mode): {e}")
        model = models.mobilenet_v2(weights=None)

    # 1. Freeze backbone parameters (Feature Extractor)
    for param in model.parameters():
        param.requires_grad = False

    # 2. Replace the final classification head for our custom dataset (2 classes: Cat vs Dog)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    print(f"Replaced final classifier: Linear(in_features={in_features}, out_features=2)")
    print(f"Trainable parameters only in final layer: {[name for name, p in model.named_parameters() if p.requires_grad]}")

    # 3. Train only the newly added classification layer
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(42)
    sample_images = torch.randn(4, 3, 224, 224)
    sample_labels = torch.tensor([0, 1, 0, 1])

    print("\nFine-tuning classifier head for 3 epochs on custom dataset...")
    model.train()
    for epoch in range(1, 4):
        optimizer.zero_grad()
        outputs = model(sample_images)
        loss = criterion(outputs, sample_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/3] - Loss: {loss.item():.4f}")

    print("[SUCCESS] Transfer learning fine-tuning completed successfully!\n")

if __name__ == "__main__":
    run_transfer_learning()

