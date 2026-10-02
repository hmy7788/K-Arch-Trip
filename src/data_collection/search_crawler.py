"""검색 엔진 이미지 크롤링 (수집 우선순위 2순위 — 네이버 쇼핑 API로 부족한 분량만 보충).

여기서 받는 이미지는 **확정 라벨이 아니다.** 검색 키워드가 노린 "의도된 클래스"일 뿐이고,
data/raw/<class>/ 아래 저장된 뒤 사람이 4종 판정 규칙(CLAUDE.md 참고)으로 눈으로
확인해서 train/{class} 또는 data/_hold로 옮겨야 한다 (docs/data-collection-plan.md 참고).

실행 예:
    python src/data_collection/search_crawler.py --class straight --max-per-keyword 50
    python src/data_collection/search_crawler.py --class all --max-per-keyword 50 --engines bing
"""

import argparse
import os
import re
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from PIL import Image, UnidentifiedImageError

# docs/data-collection-plan.md "클래스별 검색어 제안"과 동일하게 유지할 것
KEYWORDS_BY_CLASS = {
    "straight": ["일자 텀블러", "스트레이트 텀블러"],
    "taper_smooth": ["테이퍼 텀블러", "커피 텀블러"],
    "taper_step": ["스탠리 텀블러", "퀜처 텀블러"],
    "mug": ["손잡이 머그컵", "스텐 머그컵"],
}

ENGINE_CRAWLERS = {
    "google": GoogleImageCrawler,
    "bing": BingImageCrawler,
}

MIN_IMAGE_SIZE = (200, 200)  # 썸네일/아이콘 수준의 작은 이미지는 거른다
MAX_ASPECT_RATIO = 2.2  # 이보다 세로/가로(또는 가로/세로)가 길면 쇼핑몰 "상세페이지" 합성 이미지로 간주


def _slugify(keyword: str) -> str:
    return re.sub(r"\s+", "_", keyword.strip())


def _remove_extreme_aspect_ratio_images(out_dir: str) -> int:
    """한국 쇼핑몰 상세페이지처럼 세로로 아주 긴 합성 이미지를 걸러낸다.

    검증 중 실제로 발견한 문제: Bing/Google이 "일자 텀블러" 같은 키워드에도
    텀블러 실물 사진이 아니라 980x4500 같은 상세페이지 이미지를 섞어서 준다
    (docs/troubleshooting.md 기록). UI 스크린샷처럼 비율은 정상인데 내용이
    엉뚱한 경우는 이 필터로 못 거르므로, 사람 검수는 여전히 필요하다.
    """
    if not os.path.isdir(out_dir):
        return 0
    removed = 0
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        try:
            with Image.open(path) as img:
                w, h = img.size
        except (UnidentifiedImageError, OSError):
            os.remove(path)
            removed += 1
            continue
        if w == 0 or h == 0:
            continue
        ratio = max(w / h, h / w)
        if ratio > MAX_ASPECT_RATIO:
            os.remove(path)
            removed += 1
    return removed


def crawl_keyword(keyword: str, engine: str, out_dir: str, max_num: int) -> None:
    """한 키워드 x 한 엔진 조합으로 이미지를 out_dir에 받는다."""
    crawler_cls = ENGINE_CRAWLERS[engine]
    crawler = crawler_cls(
        storage={"root_dir": out_dir},
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=2,  # 과도한 동시 요청으로 차단당하지 않도록 낮게 유지
        log_level=30,  # WARNING 이상만 — icrawler 기본 로그가 너무 시끄러움
    )
    crawler.crawl(keyword=keyword, max_num=max_num, min_size=MIN_IMAGE_SIZE)
    removed = _remove_extreme_aspect_ratio_images(out_dir)
    if removed:
        print(f"  (상세페이지 추정 이미지 {removed}장 제거)")


def crawl_class(class_name: str, engines=("google", "bing"), max_per_keyword: int = 50) -> None:
    """한 클래스의 모든 키워드 x 엔진 조합을 순회한다.

    한 조합이 실패해도(엔진 차단, 페이지 구조 변경 등) 예외를 잡아서 다음
    조합으로 계속 진행한다 — 구글이 막혀도 빙 결과는 받아지도록.
    """
    keywords = KEYWORDS_BY_CLASS[class_name]
    for keyword in keywords:
        for engine in engines:
            out_dir = f"data/raw/{class_name}/{engine}_{_slugify(keyword)}"
            print(f"[{class_name}] {engine} <- \"{keyword}\" (max {max_per_keyword}) -> {out_dir}")
            try:
                crawl_keyword(keyword, engine, out_dir, max_per_keyword)
            except Exception as e:  # noqa: BLE001 - 크롤링 실패는 흔하니 계속 진행해야 함
                print(f"  !! 실패: {engine}/\"{keyword}\" — {e}")
                print("     (docs/troubleshooting.md에 기록할 것)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class", dest="class_name", required=True,
        choices=[*KEYWORDS_BY_CLASS.keys(), "all"],
    )
    parser.add_argument("--max-per-keyword", type=int, default=50)
    parser.add_argument(
        "--engines", nargs="+", default=["google", "bing"],
        choices=list(ENGINE_CRAWLERS.keys()),
    )
    args = parser.parse_args()

    targets = list(KEYWORDS_BY_CLASS.keys()) if args.class_name == "all" else [args.class_name]
    for class_name in targets:
        crawl_class(class_name, engines=tuple(args.engines), max_per_keyword=args.max_per_keyword)


if __name__ == "__main__":
    main()
