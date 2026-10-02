"""종횡비 보존 패딩 전처리 — `taehyun/ViT` 브랜치(`src/models/transforms.py`)에서 이식.

우리 ResNet 트랙은 Resize+CenterCrop을 써서 물체 일부가 잘릴 수 있는데(클로즈업
실패 사례에서 확인된 문제), 이 방식은 자르지 않고 회색 여백을 채워 물체 전체를
프레임에 담는다.
"""

import torchvision.transforms as T
from PIL import Image, ImageOps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NEUTRAL_PAD_COLOR = (128, 128, 128)


class AspectRatioPadResize:
    """물체의 종횡비를 보존하면서 target_size 정사각 캔버스 중앙에 배치하고 남는 영역을 회색으로 패딩한다."""

    def __init__(self, target_size: int = 224, fill_color=NEUTRAL_PAD_COLOR):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        scale = min(self.target_size / w, self.target_size / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.target_size, self.target_size), self.fill_color)
        canvas.paste(resized, ((self.target_size - nw) // 2, (self.target_size - nh) // 2))
        return canvas


def get_train_transforms(target_size: int = 224) -> T.Compose:
    return T.Compose([
        AspectRatioPadResize(target_size=target_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, fill=NEUTRAL_PAD_COLOR),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(target_size: int = 224) -> T.Compose:
    return T.Compose([
        AspectRatioPadResize(target_size=target_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_plain_transforms(target_size: int = 224) -> T.Compose:
    """완전 베이스라인용: 종횡비 보존 패딩·증강 없이 단순 Resize+CenterCrop만 사용
    (우리 ResNet 트랙과 동일한 방식). train/eval 둘 다 이 변환을 그대로 쓴다."""
    return T.Compose([
        T.Resize(int(target_size * 256 / 224)),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
