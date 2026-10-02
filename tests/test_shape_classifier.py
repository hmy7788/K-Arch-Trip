import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rule_based.shape_classifier import classify_shape, compute_width_profile, detect_handle
from rule_based.synthetic import (
    make_mug_mask,
    make_straight_mask,
    make_taper_smooth_mask,
    make_taper_step_mask,
)


@pytest.mark.parametrize(
    "mask_factory, expected_shape",
    [
        (make_mug_mask, "mug"),
        (make_straight_mask, "straight"),
        (make_taper_smooth_mask, "taper_smooth"),
        (make_taper_step_mask, "taper_step"),
    ],
)
def test_classify_shape_basic(mask_factory, expected_shape):
    mask = mask_factory()
    result = classify_shape(mask)
    assert result["shape"] == expected_shape


def test_taper_smooth_with_handle_stays_taper_smooth():
    """README 규칙: 실루엣이 손잡이보다 우선 -> 손잡이 달린 테이퍼는 mug가 아니라 taper_smooth."""
    mask = make_taper_smooth_mask(with_handle=True)
    result = classify_shape(mask)
    assert result["shape"] == "taper_smooth"
    assert result["shape"] != "mug"


def test_mug_with_handle_stays_mug():
    mask = make_mug_mask(with_handle=True)
    result = classify_shape(mask)
    assert result["shape"] == "mug"


def test_compute_width_profile_circle():
    mask = np.zeros((200, 200), dtype=np.uint8)
    import cv2

    cv2.circle(mask, (100, 100), 80, 255, thickness=-1)
    profile = compute_width_profile(mask, n_bins=10)
    assert len(profile) == 10
    # 원은 중앙 구간 폭이 위/아래 구간 폭보다 넓어야 한다
    assert profile[4] > profile[0]
    assert profile[4] > profile[-1]


def test_compute_width_profile_empty_mask_raises():
    mask = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_width_profile(mask)


def test_detect_handle_true_for_shape_with_bump():
    mask = make_taper_smooth_mask(with_handle=True)
    assert detect_handle(mask) is True


def test_detect_handle_false_for_plain_rectangle():
    mask = make_straight_mask(with_handle=False)
    assert detect_handle(mask) is False
