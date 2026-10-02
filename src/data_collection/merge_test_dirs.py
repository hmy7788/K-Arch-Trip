"""data/test2를 data/test에 병합한다 (파일 내용 해시로 중복 검사).

파일명이 달라도 내용이 완전히 같으면(SHA256 동일) 중복으로 간주해 건너뛴다.
src 폴더 이름이 겹치는 파일명이 있으면 접미사를 붙여 저장한다. 병합 후
data/test2는 삭제한다.

실행:
    python src/data_collection/merge_test_dirs.py --src data/test2 --dst data/test
"""

import argparse
import hashlib
import os
import shutil
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

CLASS_NAMES = ("straight", "taper_smooth", "taper_step", "mug")


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_dest_path(dst_dir: str, filename: str) -> str:
    candidate = os.path.join(dst_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    name, ext = os.path.splitext(filename)
    i = 1
    while True:
        candidate = os.path.join(dst_dir, f"{name}_dup{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def merge_class(src_root: str, dst_root: str, cls: str) -> tuple[int, int]:
    """Returns: (병합된 장수, 중복으로 건너뛴 장수)."""
    src_dir = os.path.join(src_root, cls)
    dst_dir = os.path.join(dst_root, cls)
    if not os.path.isdir(src_dir):
        return 0, 0
    os.makedirs(dst_dir, exist_ok=True)

    existing_hashes = {}
    for name in os.listdir(dst_dir):
        path = os.path.join(dst_dir, name)
        if os.path.isfile(path):
            existing_hashes[file_hash(path)] = name

    merged, skipped = 0, 0
    for name in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            continue
        h = file_hash(src_path)
        if h in existing_hashes:
            print(f"  [{cls}] 중복 건너뜀: {name} (기존 {existing_hashes[h]}와 내용 동일)")
            skipped += 1
            continue
        dst_path = unique_dest_path(dst_dir, name)
        shutil.copy2(src_path, dst_path)
        existing_hashes[h] = os.path.basename(dst_path)
        merged += 1

    return merged, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/test2")
    parser.add_argument("--dst", default="data/test")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"{args.src} 없음 — 종료")
        return

    print(f"{args.src} -> {args.dst} 병합 시작 (내용 해시로 중복 검사)\n")
    total_merged, total_skipped = 0, 0
    for cls in CLASS_NAMES:
        merged, skipped = merge_class(args.src, args.dst, cls)
        total_merged += merged
        total_skipped += skipped
        print(f"[{cls}] 병합 {merged}장, 중복 {skipped}장")

    print(f"\n합계: 병합 {total_merged}장, 중복 제외 {total_skipped}장")

    shutil.rmtree(args.src)
    print(f"{args.src} 삭제 완료")

    print(f"\n{args.dst} 클래스별 최종 장수")
    grand_total = 0
    for cls in CLASS_NAMES:
        class_dir = os.path.join(args.dst, cls)
        count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]) if os.path.isdir(class_dir) else 0
        grand_total += count
        print(f"  {cls:14s} {count:4d}장")
    print(f"  {'합계':14s} {grand_total:4d}장")


if __name__ == "__main__":
    main()
