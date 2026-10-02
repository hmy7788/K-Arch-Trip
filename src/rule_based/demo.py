"""합성 마스크 4종에 대한 classify_shape() 결과를 콘솔에 출력하는 수동 확인용 스크립트.

실행: python src/rule_based/demo.py
"""

from shape_classifier import SHAPE_LABELS_KO, classify_shape
from synthetic import make_mug_mask, make_straight_mask, make_taper_smooth_mask, make_taper_step_mask

SHAPES = {
    "머그형 (손잡이 없음)": make_mug_mask(),
    "머그형 (손잡이 있음)": make_mug_mask(with_handle=True),
    "직선 원통형": make_straight_mask(),
    "연속 테이퍼형 (손잡이 없음)": make_taper_smooth_mask(),
    "연속 테이퍼형 (손잡이 있음, 머그형으로 잘못 분류되면 안 됨)": make_taper_smooth_mask(with_handle=True),
    "단차 테이퍼형": make_taper_step_mask(),
}


def main():
    for name, mask in SHAPES.items():
        result = classify_shape(mask)
        shape_ko = SHAPE_LABELS_KO.get(result["shape"], result["shape"])
        print(f"[{name}]")
        print(f"  -> 형태={shape_ko}({result['shape']}), "
              f"높이/지름={result['height_diameter_ratio']:.2f}, "
              f"아래/위폭={result['bottom_top_ratio']:.2f}, "
              f"손잡이감지={result['handle_detected']}")


if __name__ == "__main__":
    main()
