"""새로 모은 `new_<class>/` 폴더를 기존 `<class>/` 폴더에 병합한다.

기존 폴더의 마지막 번호 다음부터 이어서 번호를 매기고, png 등 다른 포맷은
jpg로 변환한다 (normalize_raw_images.py와 동일한 정리 규칙). 병합이 끝난
new_<class> 폴더는 삭제한다.

실행:
    python src/data_collection/merge_new_images.py --root data/raw2
"""

import argparse
import glob
import os
import re
import shutil
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ("straight", "taper_smooth", "taper_step", "mug")
JPEG_QUALITY = 92


def _next_index(class_dir: str) -> int:
    """class_dir 안의 기존 파일(N.jpg) 중 가장 큰 번호 다음 번호를 반환한다."""
    max_idx = 0
    for path in glob.glob(os.path.join(class_dir, "*.jpg")):
        name = os.path.splitext(os.path.basename(path))[0]
        if re.fullmatch(r"\d+", name):
            max_idx = max(max_idx, int(name))
    return max_idx + 1


def merge_class(root: str, class_name: str) -> tuple[int, int]:
    """new_<class_name>을 <class_name>에 병합한다. Returns: (성공, 실패)."""
    src_dir = os.path.join(root, f"new_{class_name}")
    dst_dir = os.path.join(root, class_name)
    if not os.path.isdir(src_dir):
        return 0, 0

    os.makedirs(dst_dir, exist_ok=True)
    next_idx = _next_index(dst_dir)

    src_files = sorted(
        f for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f))
    )

    ok_count, failed = 0, []
    for filename in src_files:
        src_path = os.path.join(src_dir, filename)
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                dst_path = os.path.join(dst_dir, f"{next_idx}.jpg")
                img.save(dst_path, "JPEG", quality=JPEG_QUALITY)
            next_idx += 1
            ok_count += 1
        except (UnidentifiedImageError, OSError) as e:
            failed.append((filename, str(e)))

    for filename, err in failed:
        print(f"  !! 건너뜀(손상/미지원 포맷 추정): {filename} — {err}")

    shutil.rmtree(src_dir)
    return ok_count, len(failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw2")
    args = parser.parse_args()

    print(f"{'클래스':14s} {'병합':>6s} {'실패':>6s}")
    print("-" * 30)
    total = 0
    for class_name in CLASS_NAMES:
        ok, failed = merge_class(args.root, class_name)
        total += ok
        print(f"{class_name:14s} {ok:6d} {failed:6d}")
    print("-" * 30)
    print(f"{'합계':14s} {total:6d}")


if __name__ == "__main__":
    main()
