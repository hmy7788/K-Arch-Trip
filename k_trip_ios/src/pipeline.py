"""통합 추론 파이프라인 — 이미지(BGR) → 형태 라벨·확신도·클래스별 확률.

    checkpoints/model/meta.json   ← 학습한 사람한테 받은 사실 (git에 올라감)
    checkpoints/model/best.pt     ← 가중치 (git에 안 올라감, 따로 전달)

외부 의존 없이 이 파일 하나로 모델을 띄운다. 학습 코드가 어떻게 생겼든 추론에 필요한 건
넷뿐이다 — ① 라이브러리·구조 ② 입력 크기·리사이즈·정규화 ③ 클래스 순서 ④ 가중치 파일.
그 넷을 코드에 박지 않고 meta.json에 적으므로, 모델을 바꿀 때 이 파일은 안 건드린다.

값이 틀리면 에러 없이 조용히 틀린다(train/serve skew). 그래서 잡을 수 있는 건 로드 시점에 잡는다:
  - meta.json 없음/항목 누락 → 무엇을 채울지 알려주고 멈춤
  - 가중치 키가 구조와 안 맞음 → library/arch가 학습과 다르다는 뜻, 빠진·남는 키를 찍고 멈춤
  - 체크포인트에 classes가 들어있으면 meta.json과 대조

체크포인트 형식은 흔한 것을 전부 받는다:
  state_dict 그대로 / {"state_dict": …} / {"model_state_dict": …} / {"model": …} /
  DataParallel의 "module." 접두어 / torch.save(model)로 통째로 저장된 nn.Module

주의 — EXIF 회전. cv2로 읽은 폰 사진은 EXIF 회전이 적용되지 않는다. 학습이 PIL(exif_transpose)로
됐다면 회전값이 남은 사진은 다르게 누울 수 있다. 촬영 단계에서 회전을 픽셀에 굽는 것이 해법.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

RESIZE_MODES = ("stretch", "center_crop", "letterbox")
DEFAULT_MEAN = [0.485, 0.456, 0.406]   # ImageNet 통계. 다른 값을 썼으면 meta.json에서 덮어쓴다
DEFAULT_STD = [0.229, 0.224, 0.225]

REQUIRED = {
    "library": "torchvision | timm",
    "arch": "예: convnext_tiny (torchvision) / convnext_tiny.fb_in22k (timm)",
    "img_size": "학습 입력 한 변 픽셀. 예: 224",
    "resize": "stretch | center_crop | letterbox — 학습 전처리와 같아야 한다",
    "classes": "학습 때 클래스 순서. 모델 출력 0번이 이 리스트의 0번",
}


# ---------------------------------------------------------------- meta.json

def load_meta(path: Path) -> dict:
    """meta.json을 읽고 필수 항목·값을 검사한다. 없거나 틀리면 무엇을 채워야 하는지 알려주고 멈춘다."""
    path = Path(path)
    if not path.is_file():
        want = "\n".join(f"    {k}: {v}" for k, v in REQUIRED.items())
        raise FileNotFoundError(
            f"{path} 없음. 같은 폴더의 meta.example.json을 복사해 meta.json으로 만들고 아래를 채워라:\n{want}")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"{path}: 빠진 항목 {missing}. 학습한 사람한테 물어봐야 한다")
    if meta["library"] not in ("torchvision", "timm"):
        raise ValueError(f"library={meta['library']!r}, torchvision 또는 timm")
    if meta["resize"] not in RESIZE_MODES:
        raise ValueError(f"resize={meta['resize']!r}, {RESIZE_MODES} 중 하나")
    if not (isinstance(meta["classes"], list) and len(meta["classes"]) >= 2
            and all(isinstance(c, str) for c in meta["classes"])):
        raise ValueError("classes는 2개 이상 문자열 리스트")
    meta.setdefault("mean", DEFAULT_MEAN)
    meta.setdefault("std", DEFAULT_STD)
    meta.setdefault("weights", "best.pt")
    meta.setdefault("state_dict_key", None)
    meta.setdefault("letterbox_fill", [255, 255, 255])
    meta["img_size"] = int(meta["img_size"])
    return meta


# ---------------------------------------------------------------- 전처리

def preprocess(image_bgr: np.ndarray, meta: dict) -> np.ndarray:
    """BGR uint8 → 정규화된 CHW float32. 학습 전처리를 meta["resize"]로 흉내낸다.

    - stretch      : (size, size)로 그냥 늘림.            transforms.Resize((s, s))
    - center_crop  : 짧은 변 s*256/224 → 가운데 s 자름.    Resize(256)+CenterCrop(224)의 일반형
    - letterbox    : 비율 유지, 여백 채워 정사각 → s.       PIL LANCZOS (레터박스 학습 스크립트와 동일)
    """
    from PIL import Image
    size, mode = meta["img_size"], meta["resize"]
    im = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]))   # BGR → RGB
    if mode == "stretch":
        im = im.resize((size, size), Image.BILINEAR)
    elif mode == "center_crop":
        from torchvision.transforms import functional as TF, InterpolationMode
        im = TF.resize(im, int(round(size * 256 / 224)), interpolation=InterpolationMode.BILINEAR)
        im = TF.center_crop(im, [size, size])
    else:  # letterbox
        w, h = im.size
        s = size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas = Image.new("RGB", (size, size), tuple(int(v) for v in meta["letterbox_fill"]))
        canvas.paste(im.resize((nw, nh), Image.LANCZOS), ((size - nw) // 2, (size - nh) // 2))
        im = canvas
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.asarray(meta["mean"], dtype=np.float32)) / np.asarray(meta["std"], dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ---------------------------------------------------------------- 모델 구성·로드

def _strip_module(sd: dict) -> dict:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _build(meta: dict, n_classes: int):
    """meta["library"] / meta["arch"]로 빈 모델을 만든다. 마지막 층은 n_classes로."""
    import torch.nn as nn
    if meta["library"] == "timm":
        try:
            import timm
        except ImportError as e:
            raise ImportError("meta.json library=timm인데 timm이 없다. python -m pip install timm") from e
        return timm.create_model(meta["arch"], pretrained=False, num_classes=n_classes)
    from torchvision import models
    ctor = getattr(models, meta["arch"], None)
    if ctor is None:
        raise ValueError(f"torchvision.models에 {meta['arch']!r}가 없다 (예: convnext_tiny / mobilenet_v3_small)")
    m = ctor(weights=None)
    # ConvNeXt classifier[2] · MobileNetV3 classifier[3] · EfficientNet classifier[1] · ResNet fc —
    # 마지막 Linear를 찾아 바꾼다. 학습 코드가 다른 방식으로 헤드를 바꿨다면 load_state_dict에서 드러난다.
    if hasattr(m, "classifier") and isinstance(m.classifier[-1], nn.Linear):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif hasattr(m, "heads") and hasattr(m.heads, "head"):      # ViT
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
    else:
        raise ValueError(f"{meta['arch']}의 마지막 층을 못 찾았다. _build에 분기를 추가해야 한다")
    return m


def _extract_state_dict(obj, meta: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"체크포인트가 state_dict나 dict가 아니라 {type(obj).__name__}")
    sd = obj
    key = meta.get("state_dict_key")
    if key:
        sd = sd[key]
    else:
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    return _strip_module(sd)


class Pipeline:
    """meta.json + 가중치 → predict(image_bgr) -> {label, score, probs, infer_ms}."""

    def __init__(self, checkpoint_dir: Path | str):
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch가 없다. python -m pip install torch torchvision") from e
        self.dir = Path(checkpoint_dir)
        self.meta = load_meta(self.dir / "meta.json")
        self.classes = list(self.meta["classes"])
        path = self.dir / self.meta["weights"]
        if not path.is_file():
            raise FileNotFoundError(f"{path} 없음. 학습한 사람한테 받은 가중치를 여기로 복사해라")

        # 텐서만 든 파일은 weights_only=True로 안전하게. 통째로 pickle된 모델이나 numpy가 섞인
        # dict는 실패하므로 그때만 False로 다시 연다 (팀원이 준 파일만 연다는 전제).
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(obj, dict) and isinstance(obj.get("classes"), list):
            # 체크포인트에 클래스 순서가 들어있으면 meta.json과 대조 — 어긋나면 조용히 틀리므로 막는다
            if [str(c) for c in obj["classes"]] != self.classes:
                raise ValueError(f"체크포인트 classes {obj['classes']} ≠ meta.json classes {self.classes}")

        if isinstance(obj, torch.nn.Module):
            model = obj
        else:
            sd = _extract_state_dict(obj, self.meta)
            model = _build(self.meta, len(self.classes))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise ValueError(
                    "가중치 키가 모델과 안 맞는다 — meta.json의 library/arch가 학습과 다를 가능성.\n"
                    f"  빠진 키 {len(missing)}개 (예 {list(missing)[:3]})\n"
                    f"  남는 키 {len(unexpected)}개 (예 {list(unexpected)[:3]})")
        model.eval()   # BatchNorm·Dropout 추론 모드. 빼먹으면 점수가 흔들린다

        # 추론이 모든 코어를 먹으면 같은 프로세스에서 영상을 받아 디코딩하고 연결을 유지하는 쪽
        # (webrtc/aiortc)이 굶는다. 영상이 느려지다 2~3초 만에 끊긴다 — 실제로 겪은 증상이다.
        #
        # 2로 고정한다. os.cpu_count()는 논리 코어(하이퍼스레딩 포함)라 그걸 기준으로 잡으면
        # 물리 코어보다 많은 스레드를 띄워 서로 방해한다. 그리고 ConvNeXt-Tiny를 한 장씩 돌릴 때는
        # 2스레드 넘어가면 어차피 잘 안 빨라진다 — 남는 코어를 영상에 주는 편이 이득이다.
        try:
            torch.set_num_threads(max(1, min(2, (os.cpu_count() or 2) - 1)))
        except Exception:
            pass
        # GPU가 있으면 쓴다 (ConvNeXt-Tiny는 CPU에서 프레임당 100~300ms, GPU면 10ms대). 없으면 CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        # channels_last: 합성곱 커널(oneDNN/cuDNN)이 실제로 쓰는 메모리 배치. ConvNeXt처럼 합성곱이
        # 대부분인 모델은 CPU에서 눈에 띄게 빨라진다. 계산 결과는 같고 배치만 바뀐다.
        try:
            model = model.to(memory_format=torch.channels_last)
            self.channels_last = True
        except Exception:
            self.channels_last = False
        self.torch, self.model = torch, model
        self.arch = f"{self.meta['library']}:{self.meta['arch']}"
        self.size = self.meta["img_size"]
        self._gradcam = None      # 처음 explain()을 부를 때 만든다 (훅을 걸므로 하나만)

    def predict(self, image_bgr: np.ndarray) -> dict:
        torch = self.torch
        t0 = time.perf_counter()
        x = torch.from_numpy(preprocess(image_bgr, self.meta)).unsqueeze(0).to(self.device)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():   # no_grad보다 한 겹 더 끈다 (버전 추적까지)
            prob = self.model(x).softmax(1)[0].cpu().numpy().copy()   # 블록 밖에서 쓰므로 떼어낸다
        probs = {c: float(prob[i]) for i, c in enumerate(self.classes)}
        label = max(probs, key=probs.get)
        return {"label": label, "score": probs[label], "probs": probs,
                "infer_ms": (time.perf_counter() - t0) * 1000}

    def explain(self, image_bgr: np.ndarray, label: str | None = None) -> dict:
        """GradCAM. {"overlay": RGB uint8, "cam": 0~1 히트맵, "label": 설명한 클래스}.

        히트맵(cam)까지 돌려주는 이유 — 그림만으로는 '어디를 봤는지' 사람이 읽어야 한다.
        src/gradcam_reason.py가 이 숫자를 받아 근거 문장을 만든다.

        판정(predict)보다 비싸다 — 순전파에 역전파까지 한 번 더 돈다. 그래서 실시간으로 매 프레임
        돌리지 않고, 사진 한 장이나 캡처한 한 장에 대해서만 부른다.
        """
        import cv2
        from .gradcam import GradCAM, overlay_cam, target_layer
        torch = self.torch
        if self._gradcam is None:      # 훅은 한 번만 건다. 다시 만들면 훅이 쌓인다
            self._gradcam = GradCAM(self.model, target_layer(self.model))
        x = torch.from_numpy(preprocess(image_bgr, self.meta)).unsqueeze(0).to(self.device)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x.requires_grad_(True)
        if label in self.classes:
            idx = self.classes.index(label)
        else:
            with torch.no_grad():
                idx = int(self.model(x).argmax(1).item())
        cam = self._gradcam.generate(x, idx)
        size = self.meta["img_size"]
        normalized = preprocess(image_bgr, self.meta).transpose(1, 2, 0)
        rgb = np.uint8(np.clip((normalized * np.asarray(self.meta["std"]) + np.asarray(self.meta["mean"])) * 255, 0, 255).round())
        return {"overlay": overlay_cam(rgb, cam), "cam": cam, "label": self.classes[idx]}

    def safe_explain(self, image_bgr: np.ndarray, label: str | None = None):
        """(결과 dict, 오류). 히트맵을 못 그려도 판정 화면은 살아 있어야 한다."""
        try:
            return self.explain(image_bgr, label), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def safe_predict(self, image_bgr: np.ndarray) -> dict:
        """예외를 삼켜 {"pred": dict|None, "infer_ms", "error"}로. 화면이 죽지 않게."""
        t0 = time.perf_counter()
        try:
            pred = self.predict(image_bgr)
        except Exception as e:
            return {"pred": None, "infer_ms": None, "error": f"{type(e).__name__}: {e}"}
        return {"pred": pred, "infer_ms": pred["infer_ms"], "error": None}

    def info(self) -> dict:
        w = self.dir / self.meta["weights"]
        return {"arch": self.arch, "input": self.size, "resize": self.meta["resize"],
                "classes": self.classes, "weights": w.name, "device": str(self.device),
                "threads": self.torch.get_num_threads(),
                "size_mb": round(w.stat().st_size / 1e6, 1) if w.is_file() else None}
