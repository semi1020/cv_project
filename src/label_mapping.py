"""
Korean main_category -> English GroundingDINO prompt mapping.

GroundingDINO uses lowercase nouns separated by ' . ' as a single text query.
Each Korean class maps to one or more English aliases (synonyms / common names).
The first alias is the canonical label, used for canonical-only single-prompt mode.

Class selection rule (2026-05-06)
---------------------------------
Active classes are derived from Sheet2 of the source Excel:
  trash-data/경상남도 김해시_AI기반 대형생활폐기물 학습데이터_20201221.xlsx

Rule: keep only sub_categories whose count is >= 100.
A main_category is active iff at least one of its sub_categories qualifies.

Result: 54 active subs (ACTIVE_SUBS) → 31 active mains (KEEP_MAINS).
DROP_FROM_PROMPT is auto-derived: every key of KOR_TO_EN not in KEEP_MAINS.

KOR_TO_EN itself retains its full 96-class history (alias work, BOOST tuning,
substring fixes) — pruning happens at active_classes() / DROP_FROM_PROMPT
time so the dict stays a stable reference for partner code (CLIP / sub).

Cross-class substring conflicts (from alias_audit.py output) handled by
removing the offending sub-alias entirely from the dict:
  텔레비전 dropped 'tv'         (⊂ 'tv cabinet'/'tv stand')
  에어컨및온풍기 dropped 'heater' (⊂ 'water heater'/'electric heater')
  세면대(양변기) reordered to 'toilet' first, dropped 'sink' (⊂ 'kitchen sink')
  책꽂이 dropped 'bookshelf'    (also owned by 진열장)
  요(담요) dropped 'blanket'    (also owned by 이불)
"""

KOR_TO_EN: dict[str, list[str]] = {
    # Furniture - chairs / sofa / table
    "의자": ["chair", "office chair", "armchair"],
    "소파": ["sofa", "couch"],
    "식탁": ["dining table"],
    "탁자": ["coffee table", "side table"],
    "상": ["low table", "korean floor table"],
    "책상": ["desk", "writing desk"],
    "침대": ["bed"],
    "침대받침대": ["bed frame"],
    "매트리스": ["mattress"],
    # 장롱 boost null result였으므로 alias 단순화 (토큰 절약, Sprint 1 4 클래스 우선).
    "장롱": ["wardrobe", "armoire"],
    "서랍장": ["dresser", "chest of drawers"],
    "신발장": ["shoe cabinet"],
    "진열장(장식장,책장,찬장)": ["display cabinet", "bookshelf", "cupboard"],
    "TV장식장(거실장)": ["tv stand", "tv cabinet"],
    "문갑": ["low chest", "drawer chest"],
    "화장대": ["dressing table", "vanity table"],
    "책꽂이(장식장형태외)": ["book rack"],
    "옷걸이": ["coat rack", "clothes hanger stand"],
    "캐비닛": ["cabinet", "metal cabinet"],
    "파티션": ["partition", "office partition"],
    "쌀통": ["rice container", "rice bin"],  # auto-dropped (max sub 24)
    "싱크대": ["kitchen sink"],
    "싱크대장": ["kitchen cabinet"],
    "렌지대": ["stove cabinet", "range cabinet"],

    # Appliances - kitchen
    "냉장고": ["refrigerator", "fridge"],
    "전자레인지": ["microwave"],
    "가스레인지": ["gas stove", "gas range"],
    "가스오븐레인지": ["gas oven"],
    "전기밥솥": ["rice cooker"],
    "식기건조기": ["dish dryer", "dish drying rack"],
    "식기세척기": ["dishwasher"],
    "정수기": ["water purifier"],  # auto-dropped (max sub 90)
    "온수기": ["water heater"],
    "온장고": ["food warmer cabinet"],  # auto-dropped (max sub 3)

    # Appliances - laundry / cleaning
    "세탁기": ["washing machine"],
    "탈수기": ["spin dryer"],
    "건조대": ["drying rack"],
    # Sprint 1: 30k에서 14.3% recall, 521건 → 보행기 흡수. 'cleaner' 차별화.
    "청소기": ["vacuum cleaner", "upright vacuum"],

    # Appliances - climate
    # 'air conditioner' alone too weak — confused with 'electric heater' (전기난로)
    # Sprint 1: 30k에서 4.7% recall, 1,093건 → 전기난로/프린트기. 'wall ac' 형태 명시.
    "에어컨및온풍기": ["air conditioner", "wall ac unit"],
    # Sprint 1: 30k에서 2.6% recall, 1,091건 → 전기난로/프린트기/컴퓨터. 직립형 강조.
    "공기청정기및가습기": ["air purifier", "humidifier", "tower purifier"],
    "선풍기": ["electric fan"],
    "전기난로": ["electric heater"],
    "보일러": ["boiler"],

    # Electronics - AV
    # Multi-alias: 'tv' 토큰이 vision-language alignment 결정적. 'tv set' 중복 제거.
    "텔레비전": ["tv", "television", "flat screen tv"],
    "전축(오디오)": ["audio system", "stereo"],
    "전축스피커": ["speaker"],
    "카세트": ["cassette"],
    "비디오": ["vcr", "video player"],

    # Electronics - office
    "컴퓨터": ["computer", "desktop computer", "monitor"],
    "프린트기": ["printer"],
    "복사기": ["copier", "photocopier"],
    "팩시밀리": ["fax machine"],

    # Electronics - misc
    "오락기": ["arcade machine", "game machine"],
    "재봉틀": ["sewing machine"],
    "피아노": ["piano"],
    "전동안마의자": ["massage chair"],
    "의료기": ["medical device"],
    "자동판매기": ["vending machine"],

    # Mats / bedding
    "전기매트(옥,황토,온수)": ["electric mat", "heating pad"],
    "이불": ["blanket", "comforter"],
    "요(담요)": ["futon"],
    "카펫": ["carpet", "rug"],
    "장판류": ["floor mat", "vinyl flooring"],
    "대자리": ["bamboo mat"],

    # Bath / sanitary
    "세면대(양변기)": ["toilet"],
    "비데기": ["bidet"],
    "욕조": ["bathtub"],
    "욕실장식장": ["bathroom cabinet"],

    # Lighting / mirrors
    # Sprint 1: 30k에서 4.4% recall, 377건 → 형광등기구. 일반 천장등 anchor 추가.
    "실내조명등기구": ["ceiling light", "chandelier", "pendant light"],
    "형광등기구": ["fluorescent light"],
    "거울(액자형)": ["mirror"],

    # Wheeled / outdoor
    "자전거": ["bicycle"],
    "유모차": ["stroller"],
    # Sprint 1: 의자→유아용카시트 227건. 'infant'를 앞에 두어 substring 약화.
    "유아용카시트": ["infant car seat"],
    "보행기": ["baby walker"],

    # Containers / bags
    "가방": ["bag", "luggage", "suitcase"],
    "아이스박스": ["ice box", "cooler"],
    "항아리": ["ceramic jar", "earthen jar"],
    "수족관": ["aquarium", "fish tank"],

    # Decor / misc home
    "시계": ["clock", "wall clock"],
    "블라인더": ["blind", "window blind"],
    "병풍": ["folding screen"],
    "문짝": ["door"],
    "창틀": ["window frame"],
    "평상": ["wooden platform"],
    "캣타워": ["cat tower", "cat tree"],

    # Outdoor / construction
    "PVC배관류": ["pvc pipe"],
    "스티로폼": ["styrofoam"],
    "목재류": ["wood plank", "lumber"],
    "유리판": ["glass panel"],
    "건축용판넬": ["construction panel"],
    "기름탱크": ["oil tank"],
    "가정용물탱크류및정화조": ["water tank", "septic tank"],
    "간판": ["signboard"],
    "풍선간판": ["balloon signboard"],

    # Safety
    "소화기": ["fire extinguisher"],
}


# Sub_categories with CSV count >= 100 (Sheet2). Single source of truth for
# the 2026-05-06 selection rule. Strings match the CSV sub_category column
# verbatim (note literal spaces in e.g. "소파_ 3인용이상" and "컴퓨터_본 체").
# Partner CLIP code can import this set as the universe of valid sub labels.
ACTIVE_SUBS: set[str] = {
    # 의자 (3 subs)
    "의자_편의용(안락,흔들,식탁)", "의자_사무용", "의자_보조,간이",
    # 텔레비전 (2)
    "텔레비전_30인치이상", "텔레비전_30인치미만",
    # 공기청정기및가습기 (2)
    "공기청정기및가습기_높이1m미만", "공기청정기및가습기_높이1m이상",
    # 청소기 (2)
    "청소기_가정용(모든규격)", "청소기_업소용(모든규격)",
    # 시계 (1)
    "시계_벽걸이용",
    # 상 (2)
    "상_4인용미만", "상_4인용이상",
    # 에어컨및온풍기 (3)
    "에어컨및온풍기_1.0㎡이상", "에어컨및온풍기_1.0㎡미만", "에어컨및온풍기_0.5㎡미만",
    # 소파 (5)
    "소파_ 3인용이상", "소파_ 1인용", "소파_ 2인용", "소파_카우치", "소파_스툴,코너",
    # 가방 (2)
    "가방_캐리어", "가방_골프가방",
    # 실내조명등기구 (2)
    "실내조명등기구_장식용", "실내조명등기구_일반",
    # 전기밥솥 (1)
    "전기밥솥_모든규격",
    # 식탁 (1)
    "식탁_4인용이상(일반)",
    # 선풍기 (1)
    "선풍기_모든규격",
    # 진열장 (2)
    "진열장(장식장,책장,찬장)_가로90㎝미만", "진열장(장식장,책장,찬장)_가로90㎝이상",
    # 냉장고 (3)
    "냉장고_500ℓ이상", "냉장고_300ℓ미만", "냉장고_300ℓ이상",
    # 세탁기 (1)
    "세탁기_8㎏이상",
    # TV장식장 (2)
    "TV장식장(거실장)_가로90㎝이상", "TV장식장(거실장)_단순받침대",
    # 소화기 (2)
    "소화기_3.5㎏이하(약제기준)", "소화기_3.5㎏초과(약제기준)",
    # 자전거 (2)
    "자전거_성인용", "자전거_아동용",
    # 형광등기구 (2)
    "형광등기구_길이1m미만", "형광등기구_길이1m이상",
    # 피아노 (1)
    "피아노_전자피아노,풍금",
    # 오락기 (1)
    "오락기_소형",
    # 책상 (1)
    "책상_가로120㎝미만",
    # 전자레인지 (1)
    "전자레인지_모든규격",
    # 세면대(양변기) (1)
    "세면대(양변기)_모든규격",
    # 프린트기 (1)
    "프린트기_업체용대형외모든규격",
    # 의료기 (1)
    "의료기_일반",
    # 컴퓨터 (2)
    "컴퓨터_모니터", "컴퓨터_본 체",
    # 거울 (2)
    "거울(액자형)_1㎡미만", "거울(액자형)_1㎡이상",
    # 옷걸이 (1)
    "옷걸이_모든규격",
    # 화장대 (1)
    "화장대_가로90㎝미만",
}

# Main categories with at least one ACTIVE sub. Derived from ACTIVE_SUBS.
KEEP_MAINS: set[str] = {s.split("_", 1)[0] for s in ACTIVE_SUBS}


# Classes whose ALL aliases are added to the prompt (not just the canonical),
# to strengthen multi-token match score. Use sparingly — each extra alias
# eats BERT tokens. Justified for classes that lose to longer cross-class
# phrases (e.g. 'chair' losing to 'massage chair').
# Only classes still in KEEP_MAINS can boost; entries outside it are filtered.
PROMPT_BOOST: set[str] = {
    "의자",              # 검증됨 (50장 0% → 53%)
    "텔레비전",          # 검증됨 (50장 0% → 50%, 30k 64%)
    # Sprint 1 추가 — 30k 거대 실패 4 클래스 회복:
    "에어컨및온풍기",    # 4.7% → 목표 ≥30%
    "공기청정기및가습기",  # 2.6% → 목표 ≥30%
    "청소기",            # 14.3% → 목표 ≥50%
    "실내조명등기구",    # 4.4% → 목표 ≥30%
    # 장롱은 KEEP_MAINS에서 제외되어 자동 비활성 (소분류 max 51 < 100)
}


# Classes excluded from the GDINO prompt — auto-derived from KEEP_MAINS.
# Anything in KOR_TO_EN that is not an active main gets dropped.
# This includes:
#   - mains whose all sub_categories have count < 100 (new 2026-05-06 rule)
#   - mains absent from the CSV (병풍, 기름탱크)
DROP_FROM_PROMPT: set[str] = {kor for kor in KOR_TO_EN if kor not in KEEP_MAINS}


# Build flat alias -> canonical Korean label mapping (for lookup after detection)
EN_ALIAS_TO_KOR: dict[str, str] = {}
for kor, aliases in KOR_TO_EN.items():
    for a in aliases:
        EN_ALIAS_TO_KOR[a.lower()] = kor


def active_classes() -> list[str]:
    """Korean classes that participate in the GDINO prompt (DROP_FROM_PROMPT removed)."""
    return [k for k in KOR_TO_EN if k not in DROP_FROM_PROMPT]


def build_gdino_text_prompt(canonical_only: bool = True) -> str:
    """Build a single GDINO text prompt.

    canonical_only=True (default): one alias per active class — except classes
    in PROMPT_BOOST, which always contribute all aliases. Designed to fit the
    BERT 256-token limit while giving classes that lose to longer cross-class
    phrases extra signal.

    canonical_only=False: all aliases for every class (deduped, dict order).
    May exceed the token limit; use only with chunked inference.

    GDINO format: lowercase phrases separated by ' . '
    """
    seen: set[str] = set()
    parts: list[str] = []
    for kor in active_classes():
        aliases = KOR_TO_EN[kor]
        if kor in PROMPT_BOOST:
            chosen = aliases
        else:
            chosen = aliases[:1] if canonical_only else aliases
        for a in chosen:
            a = a.lower().strip()
            if a not in seen:
                seen.add(a)
                parts.append(a)
    return " . ".join(parts) + " ."


def map_en_to_kor(en_label: str) -> str | None:
    """Map a detected English label back to the canonical Korean main_category.

    GDINO sometimes returns multi-word phrases or partial matches; we fall back
    to substring matching against known aliases.
    """
    s = en_label.lower().strip()
    if s in EN_ALIAS_TO_KOR:
        return EN_ALIAS_TO_KOR[s]
    # substring fallback: longest alias contained in s wins
    best = None
    best_len = 0
    for alias, kor in EN_ALIAS_TO_KOR.items():
        if alias in s and len(alias) > best_len:
            best = kor
            best_len = len(alias)
    return best


if __name__ == "__main__":
    print(f"Korean classes: {len(KOR_TO_EN)}")
    print(f"English aliases: {len(EN_ALIAS_TO_KOR)}")
    prompt = build_gdino_text_prompt()
    print(f"Prompt length: {len(prompt)} chars, {prompt.count('.')} segments")
    print(f"\nFirst 300 chars of prompt:\n{prompt[:300]}...")
