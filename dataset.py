import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=16):
        self.seq_len = seq_len
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        for label, cls in enumerate(["normal", "anomaly"]):
            class_path = os.path.join(root_dir, cls)
            for video in os.listdir(class_path):
                frames = sorted(glob.glob(os.path.join(class_path, video, "*.png")))
                if len(frames) >= seq_len:
                    self.samples.append((frames, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        frames = frames[:self.seq_len]

        images = [self.transform(Image.open(f).convert("RGB")) for f in frames]
        images = torch.stack(images)
        return images, torch.tensor(label, dtype=torch.float32)
