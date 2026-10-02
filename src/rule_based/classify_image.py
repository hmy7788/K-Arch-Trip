"""실제 이미지 파일 한 장을 룰베이스 파이프라인(get_mask -> classify_shape)으로 분류한다.

실행: python src/rule_based/classify_image.py <이미지 경로>

get_mask()는 Otsu 임계값 기반의 단순한 세그멘테이션이라 배경이 단일 색이고
대비가 뚜렷할 때만 안정적으로 동작한다 (알려진 한계, README 참고). 분류 결과가
이상하면 가장 먼저 저장된 *_mask.png를 열어 마스크가 텀블러 실루엣을 제대로
잡았는지부터 확인할 것.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from shape_classifier import classify_shape, describe_result_ko, get_mask


def imread_unicode(path: str):
    """cv2.imread는 Windows에서 경로에 non-ASCII 문자(한글 등)가 있으면 조용히
    실패한다 — 이 저장소 경로 자체가 한글(예: 사용자 폴더명)이라 실사용에서
    반드시 걸리는 문제라 np.fromfile + cv2.imdecode로 우회한다.
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True


def main():
    if len(sys.argv) != 2:
        print("사용법: python src/rule_based/classify_image.py <이미지 경로>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"파일이 존재하지 않습니다: {image_path}")
        sys.exit(1)

    image = imread_unicode(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다 (파일 손상 또는 지원하지 않는 형식): {image_path}")
        sys.exit(1)

    mask = get_mask(image)
    if not mask.any():
        print("마스크가 비어 있습니다 — 배경 분리에 실패했을 가능성이 높습니다.")
        sys.exit(1)

    mask_path = os.path.splitext(image_path)[0] + "_mask.png"
    if imwrite_unicode(mask_path, mask):
        print(f"추출된 마스크 저장: {mask_path}  (배경 분리가 잘 됐는지 먼저 눈으로 확인)")
    else:
        print(f"경고: 마스크 저장 실패 ({mask_path}) — 분류는 계속 진행합니다.")

    result = classify_shape(mask)
    print("\n[분류 결과]")
    print(describe_result_ko(result))


if __name__ == "__main__":
    main()
