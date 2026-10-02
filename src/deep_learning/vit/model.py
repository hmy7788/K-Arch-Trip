"""DeiT-Small/16 모델 정의, 가중치 동결/해제, 차등 학습률 파라미터 그룹화.

`taehyun/ViT` 브랜치(`src/models/vit_classifier.py`)의 핵심 로직을 이식한 것 —
모델 자체와 freeze/unfreeze 전략은 그대로 가져오되, 데이터 로딩은 이 프로젝트의
다른 트랙들과 동일하게 data/preprocess 기반으로 바꿔서 쓴다(train.py 참고).
"""

import timm
import torch.nn as nn

CLASSES = ["straight", "taper_smooth", "taper_step", "mug"]
DEFAULT_MODEL_NAME = "deit_small_patch16_224.fb_in1k"


def create_deit_model(model_name: str = DEFAULT_MODEL_NAME, num_classes: int = len(CLASSES),
                       pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    """timm의 DeiT-Small/16 사전학습 모델을 로드하고 분류 헤드를 4클래스로 교체한다."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def unfreeze_all(model: nn.Module) -> None:
    """완전 베이스라인용: 부분 선택 없이 전체 파라미터를 그냥 다 학습 가능하게 둔다(기본 상태)."""
    for param in model.parameters():
        param.requires_grad = True


def freeze_backbone_for_linear_probe(model: nn.Module) -> None:
    """실험 A: 백본 전체를 동결하고 분류 헤드만 학습 가능하게 한다."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def unfreeze_last_blocks_for_fine_tuning(model: nn.Module, num_blocks: int = 4) -> None:
    """실험 B: 마지막 num_blocks(기본 4개) Transformer 블록 + 최종 LayerNorm + 분류 헤드만 학습 가능하게 한다."""
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "blocks"):
        for block in model.blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
    if hasattr(model, "norm") and model.norm is not None:
        for param in model.norm.parameters():
            param.requires_grad = True
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def get_optimizer_param_groups(model: nn.Module, lr_blocks: float = 1e-5, lr_head: float = 1e-4,
                                weight_decay: float = 0.01) -> list:
    """실험 B용 차등 학습률(백본 블록 vs 분류 헤드) 파라미터 그룹을 만든다."""
    head_params, block_params = [], []
    head_param_ids = set(id(p) for p in model.get_classifier().parameters())
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (head_params if id(p) in head_param_ids else block_params).append(p)

    groups = []
    if block_params:
        groups.append({"params": block_params, "lr": lr_blocks, "weight_decay": weight_decay, "name": "blocks_and_norm"})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "classifier_head"})
    return groups
