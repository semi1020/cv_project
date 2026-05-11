# src/prompt_chunks.py
"""
[DEPRECATED — 2026-05-10] 청킹 모드는 v6_chunked_3g 실험에서 폐기됨.

폐기 사유:
  v3_baseline (단일 프롬프트)  : main_acc 57.0%, 4643s
  v6_chunked_3g (3그룹 청킹)   : main_acc 53.5% (-3.5pp), 10042s (2.2x slower)
  - 그룹별 GDINO score는 prompt context-dependent라 그룹 간 비교가 비공정
  - 같은 그룹에 묶인 혼동 페어(의자/소파/상)는 오히려 confusion 폭증
  - 손실이 큰 클래스: 청소기 -31.6pp, 선풍기 -22.8pp, 형광등기구 -20.0pp

본 파일은 재현/참조용으로 보관. 활성 파이프라인에서는 호출하지 않음.
"""
from src.label_mapping import KOR_TO_EN, PROMPT_BOOST, DROP_FROM_PROMPT, KEEP_MAINS

CHUNK_GROUPS = {
    "A_furniture": [
        '의자', '소파', '식탁', '책상', '화장대',
        '진열장(장식장,책장,찬장)', 'TV장식장(거실장)',
        '옷걸이', '오락기', '피아노', '거울(액자형)',
    ],
    "B_appliance_tall": [
        '에어컨및온풍기', '냉장고', '세탁기', '청소기',
        '선풍기', '자전거', '세면대(양변기)', '소화기',
        '의료기', '프린트기',
    ],
    "C_appliance_small": [
        '공기청정기및가습기', '텔레비전', '상', '컴퓨터',
        '전자레인지', '전기밥솥', '실내조명등기구',
        '형광등기구', '시계', '가방',
    ],
}


def build_chunk_prompts() -> dict[str, str]:
    """각 그룹별 GDINO 프롬프트 생성."""
    prompts = {}
    for group_name, classes in CHUNK_GROUPS.items():
        seen = set()
        parts = []
        for kor in sorted(classes):
            if kor in DROP_FROM_PROMPT:
                continue
            aliases = KOR_TO_EN[kor]
            if kor in PROMPT_BOOST:
                chosen = aliases
            else:
                chosen = aliases[:1]
            for a in chosen:
                a = a.lower().strip()
                if a not in seen:
                    seen.add(a)
                    parts.append(a)
        prompts[group_name] = " . ".join(parts) + " ."
    return prompts


def validate_chunks():
    """모든 KEEP_MAINS가 CHUNK에 포함되어 있는지 확인."""
    all_classes = set()
    for classes in CHUNK_GROUPS.values():
        all_classes.update(classes)

    missing = KEEP_MAINS - all_classes
    extra = all_classes - KEEP_MAINS
    if missing:
        print(f"[ERROR] KEEP_MAINS에 있지만 CHUNK에 없음: {missing}")
    if extra:
        print(f"[WARN] CHUNK에 있지만 KEEP_MAINS에 없음: {extra}")
    if not missing and not extra:
        print(f"[OK] 31개 클래스 전부 배치됨 ({len(CHUNK_GROUPS)}그룹)")

    prompts = build_chunk_prompts()
    for name, prompt in prompts.items():
        segs = prompt.count('.')
        print(f"  {name}: {segs} segments, {len(prompt)} chars")


if __name__ == "__main__":
    validate_chunks()
