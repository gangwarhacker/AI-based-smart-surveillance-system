import cv2
import torch
import os
from PIL import Image
from torchvision import transforms
from model import AnomalyNet

# Create alert folder
os.makedirs("alerts", exist_ok=True)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load trained model
model = AnomalyNet().to(device)
model.load_state_dict(torch.load("anomaly_model.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

sequence = []
SEQ_LEN = 16
alert_count = 0

cap = cv2.VideoCapture(0)  # Webcam

print("📡 Smart Surveillance Running... Press Q to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img)
    sequence.append(tensor)

    if len(sequence) == SEQ_LEN:
        clip = torch.stack(sequence).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(clip).item()

        label = "ANOMALY" if score > 0.5 else "NORMAL"
        color = (0,0,255) if label == "ANOMALY" else (0,255,0)

        cv2.putText(frame, f"{label} ({score:.2f})", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 🚨 ALERT SYSTEM
        if score > 0.5:
            alert_count += 1
            filename = f"alerts/alert_{alert_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"⚠️ ALERT! Suspicious activity detected. Saved: {filename}")

        sequence.pop(0)

    cv2.imshow("Smart Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
