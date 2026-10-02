"""직접 설계한 작은 CNN — 구현 공부/실험 목적의 트랙.

4트랙(룰베이스/Mask R-CNN/ResNet/EfficientNet) 공식 비교에는 포함하지 않는다
(CLAUDE.md 참고). 사전학습 없이 data/preprocess(514장 train)만으로 처음부터
학습하므로, ResNet 전이학습 대비 정확도가 얼마나 낮게 나오는지 자체가
"왜 전이학습이 유리한가"를 확인하는 대조군 실험이다.

conv block 4개(채널 32->64->128->256) + Global Average Pooling + Dropout + FC.
GAP을 쓰는 이유: 입력 해상도에 파라미터 수가 안 묶여서 224 대신 더 작은
128x128을 써도(작은 데이터셋에 유리, 학습도 빠름) 구조를 안 바꿔도 됨.
"""

import torch.nn as nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 3, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
