"""수집한 원본 이미지를 클래스 폴더별로 정리한다.

- 파일명을 1.jpg, 2.jpg, ... 순서로 통일
- png/webp 등 다른 포맷은 jpg로 변환
- 처리 후 클래스별 장수를 출력

**라벨을 확정하는 게 아니다** — 아직 사람이 4종 규칙으로 검수하기 전, 파일
정리 단계일 뿐이다 (docs/data-collection-plan.md 참고).

실행:
    python src/data_collection/normalize_raw_images.py --root data/raw2
"""

import argparse
import os
import shutil
import sys
import tempfile

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ("straight", "taper_smooth", "taper_step", "mug")
JPEG_QUALITY = 92


def normalize_class_dir(class_dir: str) -> tuple[int, int]:
    """class_dir 안의 이미지를 1.jpg, 2.jpg, ...로 재정리한다.

    변환은 임시 폴더에서 먼저 끝낸 뒤 원본과 통째로 교체한다 — 도중에
    실패해도 원본 폴더는 그대로 남아있도록 하기 위함(데이터 손실 방지).

    Returns: (성공 처리한 장수, 실패(손상 등으로 건너뛴) 장수)
    """
    if not os.path.isdir(class_dir):
        return 0, 0

    filenames = sorted(
        f for f in os.listdir(class_dir)
        if os.path.isfile(os.path.join(class_dir, f))
    )

    tmp_dir = tempfile.mkdtemp(prefix="normalize_", dir=class_dir)
    ok_count = 0
    failed = []

    try:
        for filename in filenames:
            src_path = os.path.join(class_dir, filename)
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    dst_path = os.path.join(tmp_dir, f"{ok_count + 1}.jpg")
                    img.save(dst_path, "JPEG", quality=JPEG_QUALITY)
                ok_count += 1
            except (UnidentifiedImageError, OSError) as e:
                failed.append((filename, str(e)))

        # 임시 폴더가 성공적으로 채워졌으면 원본을 지우고 교체
        for filename in filenames:
            os.remove(os.path.join(class_dir, filename))
        for name in os.listdir(tmp_dir):
            shutil.move(os.path.join(tmp_dir, name), os.path.join(class_dir, name))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    for filename, err in failed:
        print(f"  !! 건너뜀(손상/미지원 포맷 추정): {filename} — {err}")

    return ok_count, len(failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw2")
    args = parser.parse_args()

    print(f"{'클래스':14s} {'처리':>6s} {'실패':>6s}")
    print("-" * 30)
    total = 0
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(args.root, class_name)
        ok, failed = normalize_class_dir(class_dir)
        total += ok
        print(f"{class_name:14s} {ok:6d} {failed:6d}")
    print("-" * 30)
    print(f"{'합계':14s} {total:6d}")


if __name__ == "__main__":
    main()
