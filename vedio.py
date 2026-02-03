import os
import shutil

base = "raw_dataset"

for category in os.listdir(base):
    cat_path = os.path.join(base, category)
    if not os.path.isdir(cat_path):
        continue

    for file in os.listdir(cat_path):
        if not file.endswith(".png"):
            continue

        # Remove frame number → get video name
        video_name = "_".join(file.split("_")[:-1])

        video_folder = os.path.join(cat_path, video_name)
        os.makedirs(video_folder, exist_ok=True)

        shutil.move(os.path.join(cat_path, file),
                    os.path.join(video_folder, file))

print("✅ Frames grouped into video folders!")
