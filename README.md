# cv_project — 대형폐기물 분류 실험

## 프로젝트 구조

```
cv_project/
├── config.py               ← 카테고리/프롬프트 설정 (추가/제외는 여기서)
├── 00_make_split.py        ← Step 0: 1회 실행, splits/splits.json 생성
├── 01_extract_crops.py     ← Step 1: 1회 실행, splits/crop_splits.json + data/crops/
├── exp1_zeroshot.py        ← 실험 1: CLIP only vs DINO+CLIP 비교
├── exp2_train.py           ← 실험 2: Linear Probe 학습
├── exp2_evaluate.py        ← 실험 2: 평가 + zero-shot 비교
├── src/
│   ├── dataset.py          ← CSV 로딩, SampleRecord, split 유틸
│   ├── dino.py             ← Grounding DINO wrapper
│   ├── clip_zeroshot.py    ← CLIP zero-shot classifier
│   ├── linear_probe.py     ← CLIP + Linear head 모델
│   └── metrics.py          ← 평가 지표 (top1/3, macro_f1, abstain 등)
├── splits/                 ← 고정 split 저장소
├── results/exp1/ exp2/     ← 실험 결과
├── legacy/                 ← 구 코드 보관
└── requirements.txt
```

---

## 실행 순서

데이터 준비 후 (`data/trash-data/csv/`, `data/trash-data/image/`) 아래 순서로 실행합니다.

### Step 0 — Split 생성 (1회)

```bash
python 00_make_split.py
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data-root` | `data/trash-data` | CSV·이미지 루트 경로 |
| `--out` | `splits/splits.json` | 출력 경로 |
| `--train` | `0.7` | train 비율 |
| `--val` | `0.15` | val 비율 (나머지는 test) |
| `--seed` | `42` | 랜덤 시드 |
| `--max-per-category` | `100` | **sub_category별 최대 샘플 수** (0이면 전체 사용) |

`--max-per-category` 사용 예시:

```bash
# sub_category당 최대 200개
python 00_make_split.py --max-per-category 200

# 제한 없이 전체 데이터 사용
python 00_make_split.py --max-per-category 0
```

> `splits/splits.json`이 이미 존재하면 경고 후 종료됩니다.
> 재생성이 필요하면 파일을 삭제한 뒤 실행하세요.

### Step 1 — DINO Crop 추출 (1회)

```bash
python 01_extract_crops.py
```

`splits/splits.json`을 읽어 각 이미지에 Grounding DINO를 실행하고,
`data/crops/{split}/{file_name}` 에 crop을 저장한 뒤
`splits/crop_splits.json`을 생성합니다.
검출 실패 시 원본 이미지 전체를 fallback으로 사용합니다.

---

### 실험 1 — Zero-Shot 비교

```bash
python exp1_zeroshot.py --split test
```

- **CLIP only**: 원본 이미지(`splits/splits.json`) 기반 zero-shot
- **DINO + CLIP**: crop 이미지(`splits/crop_splits.json`) 기반 zero-shot
- 결과: `results/exp1/clip_only.json`, `dino_clip.json`, `comparison.json`

---

### 실험 2 — Linear Probe 학습·평가

```bash
# 학습
python exp2_train.py

# 평가 + zero-shot 비교
python exp2_evaluate.py --checkpoint runs/exp2/<run_id>/best.pt
```

- 결과: `results/exp2/linear_probe.json`, `comparison.json`

---

## 카테고리 추가/제외 방법

`config.py`의 `CATEGORY_CONFIG` 딕셔너리를 수정합니다.

- **제외**: 해당 블록을 주석 처리
- **포함**: 주석 해제 또는 새 키 추가

```python
CATEGORY_CONFIG: dict[str, dict] = {
    "소파": {
        "dino_prompt": "sofa . couch . armchair",   # Grounding DINO 프롬프트
        "sub_categories": {
            "소파_1인용": "a small single-seat armchair ...",  # CLIP 분류 텍스트
            "소파_2인용": "a medium two-seat sofa ...",
            ...
        },
    },
    # 아래 블록의 주석을 해제하면 침대 카테고리가 활성화됩니다.
    # "침대": { ... },
}
```

**주의사항**

- `sub_categories`의 키는 CSV `sub_category` 컬럼 값을 정규화한 것과 정확히 일치해야 합니다.
  - 공백 제거, `cm` → `㎝` 변환이 자동 적용됩니다.
  - 예: `"소파_ 1인용"` → `"소파_1인용"`, `"가로90cm미만"` → `"가로90㎝미만"`
- 카테고리를 변경한 경우 `splits/splits.json`을 삭제하고 Step 0부터 재실행해야 합니다.

---

## 데이터 소스

CSV 파일 위치: `/data/trash-data/csv/`

| 파일 | 설명 |
|---|---|
| `ipcamp_5868_2020-12-21.csv` | 소파, 텔레비전, 의자 등 |
| `ipcamp_5995_2020-12-21.csv` | 텔레비전, 상, 피아노 등 |
| `ipcamp_5996_2020-12-21.csv` | 프린트기, 조명, 소화기 등 |
| `ipcamp_5997_2020-12-21.csv` | 전동안마의자, 정수기, 유모차 등 |

각 CSV는 `file_name`, `main_category`, `sub_category` 3개 컬럼을 가집니다.
# cv_project
