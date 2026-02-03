import torch.nn as nn
import torchvision.models as models

class AnomalyNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b*t, c, h, w)
        feats = self.cnn(x).view(b, t, -1)
        lstm_out, _ = self.lstm(feats)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)
