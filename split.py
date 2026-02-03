import os
import shutil
import random

source = "raw_dataset"
dest = "UCF-Dataset"
train_ratio = 0.8

for cls in os.listdir(source):
    class_path = os.path.join(source, cls)
    if not os.path.isdir(class_path):
        continue

    label = "normal" if cls == "NormalVideos" else "anomaly"

    for video_folder in os.listdir(class_path):
        src_path = os.path.join(class_path, video_folder)
        if not os.path.isdir(src_path):
            continue

        split = "train" if random.random() < train_ratio else "test"
        dest_path = os.path.join(dest, split, label, video_folder)

        shutil.copytree(src_path, dest_path)

print("✅ Dataset ready for ML!")

