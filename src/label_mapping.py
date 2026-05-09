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

Prompt v3 changes (2026-05-08)
------------------------------
v2 (same day) caused 식탁 to absorb chair/table/TV via 'large dining table
with legs', 공기청정기 to absorb 소화기/밥솥 via 'cylindrical air cleaner',
실내조명 to absorb 선풍기 via 'round ceiling lamp'.

v3 reverts those three aliases, keeps all other v2 improvements:
  식탁: 'large dining table with legs' → 'dining table set' (less aggressive)
  공기청정기: 'cylindrical air cleaner' → 'tower air purifier' (less generic)
  실내조명: 'round ceiling lamp' → 'pendant light' (restore v1)
  나머지 v2 변경 유지 (화장대, 에어컨, TV장식장, 책상, 의료기, 소파, 옷걸이, 오락기, 형광등)
"""

KOR_TO_EN: dict[str, list[str]] = {
    # Furniture - chairs / sofa / table
    '의자': ['chair', 'office chair', 'armchair'],
    '소파': ['upholstered sofa', 'large padded couch'],
    "식탁": ["dining table", "dining table set"],                           # ★ v3: 'large dining table with legs' → 'dining table set'
    "탁자": ["coffee table", "side table"],
    '상': ['low table', 'small folding table'],
    "책상": ["desk", "office desk", "computer desk"],                       # ★ v2 유지
    "침대": ["bed"],
    "침대받침대": ["bed frame"],
    "매트리스": ["mattress"],
    "장롱": ["wardrobe", "armoire"],
    "서랍장": ["dresser", "chest of drawers"],
    "신발장": ["shoe cabinet"],
    "진열장(장식장,책장,찬장)": ["display cabinet", "bookshelf", "cupboard"],
    "TV장식장(거실장)": ["tv stand", "media cabinet"],                      # ★ v2 유지
    "문갑": ["low chest", "drawer chest"],
    "화장대": ["vanity with mirror", "dressing table with mirror"],         # ★ v2 유지
    "책꽂이(장식장형태외)": ["book rack"],
    "옷걸이": ["coat rack", "standing coat rack"],                          # ★ v2 유지
    "캐비닛": ["cabinet", "metal cabinet"],
    "파티션": ["partition", "office partition"],
    "쌀통": ["rice container", "rice bin"],
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
    "정수기": ["water purifier"],
    "온수기": ["water heater"],
    "온장고": ["food warmer cabinet"],

    # Appliances - laundry / cleaning
    "세탁기": ["washing machine"],
    "탈수기": ["spin dryer"],
    "건조대": ["drying rack"],
    "청소기": ["vacuum cleaner", "upright vacuum"],

    # Appliances - climate
    '에어컨및온풍기': ['air conditioner', 'wall ac unit', 'standing air conditioner'],
    '공기청정기및가습기': ['air purifier', 'humidifier', 'tower air purifier', 'round top air purifier'],
    "선풍기": ["electric fan"],
    "전기난로": ["electric heater"],
    "보일러": ["boiler"],

    # Electronics - AV
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
    "오락기": ["arcade cabinet", "coin operated game machine"],             # ★ v2 유지
    "재봉틀": ["sewing machine"],
    "피아노": ["piano"],
    "전동안마의자": ["massage chair"],
    "의료기": ["wheelchair", "medical walker", "mobility aid"],             # ★ v2 유지
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
    "실내조명등기구": ["ceiling light", "chandelier", "pendant light"],      # ★ v3: 'round ceiling lamp' → 'pendant light' (v1 복원)
    "형광등기구": ["fluorescent tube light", "long fluorescent lamp"],       # ★ v2 유지
    "거울(액자형)": ["mirror"],

    # Wheeled / outdoor
    "자전거": ["bicycle"],
    "유모차": ["stroller"],
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


ACTIVE_SUBS: set[str] = {
    "의자_편의용(안락,흔들,식탁)", "의자_사무용", "의자_보조,간이",
    "텔레비전_30인치이상", "텔레비전_30인치미만",
    "공기청정기및가습기_높이1m미만", "공기청정기및가습기_높이1m이상",
    "청소기_가정용(모든규격)", "청소기_업소용(모든규격)",
    "시계_벽걸이용",
    "상_4인용미만", "상_4인용이상",
    "에어컨및온풍기_1.0㎡이상", "에어컨및온풍기_1.0㎡미만", "에어컨및온풍기_0.5㎡미만",
    "소파_3인용이상", "소파_1인용", "소파_2인용", "소파_카우치", "소파_스툴,코너",
    "가방_캐리어", "가방_골프가방",
    "실내조명등기구_장식용", "실내조명등기구_일반",
    "전기밥솥_모든규격",
    "식탁_4인용이상(일반)",
    "선풍기_모든규격",
    "진열장(장식장,책장,찬장)_가로90㎝미만", "진열장(장식장,책장,찬장)_가로90㎝이상",
    "냉장고_500ℓ이상", "냉장고_300ℓ미만", "냉장고_300ℓ이상",
    "세탁기_8㎏이상",
    "TV장식장(거실장)_가로90㎝이상", "TV장식장(거실장)_단순받침대",
    "소화기_3.5㎏이하(약제기준)", "소화기_3.5㎏초과(약제기준)",
    "자전거_성인용", "자전거_아동용",
    "형광등기구_길이1m미만", "형광등기구_길이1m이상",
    "피아노_전자피아노,풍금",
    "오락기_소형",
    "책상_가로120㎝미만",
    "전자레인지_모든규격",
    "세면대(양변기)_모든규격",
    "프린트기_업체용대형외모든규격",
    "의료기_일반",
    "컴퓨터_모니터", "컴퓨터_본체",
    "거울(액자형)_1㎡미만", "거울(액자형)_1㎡이상",
    "옷걸이_모든규격",
    "화장대_가로90㎝미만",
}

KEEP_MAINS: set[str] = {s.rsplit("_", 1)[0] for s in ACTIVE_SUBS}


PROMPT_BOOST = {
    '의자',
    '텔레비전',
    '에어컨및온풍기',
    '공기청정기및가습기',
    '청소기',
    '실내조명등기구',
    '화장대',
    '형광등기구',
}

DROP_FROM_PROMPT: set[str] = {kor for kor in KOR_TO_EN if kor not in KEEP_MAINS}


EN_ALIAS_TO_KOR: dict[str, str] = {}
for kor, aliases in KOR_TO_EN.items():
    for a in aliases:
        EN_ALIAS_TO_KOR[a.lower()] = kor


def active_classes() -> list[str]:
    return sorted(k for k in KOR_TO_EN if k not in DROP_FROM_PROMPT)


def build_gdino_text_prompt(canonical_only: bool = True) -> str:
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
    s = en_label.lower().strip()
    if s in EN_ALIAS_TO_KOR:
        kor = EN_ALIAS_TO_KOR[s]
        if kor in DROP_FROM_PROMPT:
            return None
        return kor
    best = None
    best_len = 0
    for alias, kor in EN_ALIAS_TO_KOR.items():
        if kor in DROP_FROM_PROMPT:
            continue
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
