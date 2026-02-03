import torch
from torch.utils.data import DataLoader
from dataset import VideoDataset
from model import AnomalyNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_data = VideoDataset("UCF-Dataset/train")
test_data = VideoDataset("UCF-Dataset/test")

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4)

model = AnomalyNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for clips, labels in tqdm(train_loader):
        clips, labels = clips.to(device), labels.to(device).unsqueeze(1)
        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "anomaly_model.pth")
print("✅ Model saved!")

# Test accuracy
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for clips, labels in test_loader:
        clips, labels = clips.to(device), labels.to(device).unsqueeze(1)
        outputs = model(clips)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Test Accuracy:", correct/total)
