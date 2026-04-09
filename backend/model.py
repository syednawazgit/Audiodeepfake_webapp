import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """
    Matches `best_wav2vec_lfcc_model.pth` and testaudio.py FusionModel:
    LFCC CNN with MaxPool after conv1/2, GAP, concat Wav2Vec2 (768) → 256 → 1.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((1, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128 + 768, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, lfcc, wav2vec):
        # testaudio passes (B, 20, T); inference passes (B, 1, 20, T)
        x = lfcc.unsqueeze(1) if lfcc.dim() == 3 else lfcc

        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        fused = torch.cat([x, wav2vec], dim=1)
        x = self.dropout(torch.relu(self.fc1(fused)))
        return self.fc2(x)


# Backward-compatible name for imports
FusionResNet = FusionModel
