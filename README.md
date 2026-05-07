# cv_project — 김해시 대형폐기물 분류

이미지 1장 → main_category(31종) + sub_category(54종) 자동 분류 → 김해시 배출수수료 안내.

## 시스템 개요 (2-stage)

```
                 ┌──────────────────────────────┐
                 │        E2E inference         │
                 │   10_e2e_pipeline.py         │
                 └──────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
    [Stage A] GDINO            [Stage B] GDINO + CLIP
    single-prompt (31)          per-class crop → sub
    main_category 예측           main별 dino_prompt
                                 → CLIP sub 분류
```

학습/평가 흐름은 GT main을 가정한 별도 경로(`01_extract_crops.py` → `02_dino_eval.py` → `exp1`/`exp2`)를 사용합니다.

## 프로젝트 구조

```
cv_project/
├── 00_make_split.py        Step 0: splits/splits.json
├── 01_extract_crops.py     Step 1: splits/crop_splits.json + data/crops/
├── 02_dino_eval.py         Step 2: results/dino_eval/ (BBOX 평가)        [신규]
├── 10_e2e_pipeline.py      E2E   : outputs/e2e_predictions.jsonl        [신규]
├── exp1_zeroshot.py        실험 1: CLIP only vs DINO+CLIP
├── exp2_train.py           실험 2: Linear Probe 학습
├── exp2_evaluate.py        실험 2: 평가
├── config.py               자동 생성 — 직접 편집 금지
├── tools/
│   └── sync_label_mapping.py   src/label_mapping.py → config.py 변환    [신규]
├── src/
│   ├── label_mapping.py    KOR_TO_EN, ACTIVE_SUBS, KEEP_MAINS (단일 소스)[신규]
│   ├── dataset.py          SampleRecord(+dino_meta), split 유틸
│   ├── dino.py             Grounding DINO base wrapper
│   ├── clip_zeroshot.py    CLIP zero-shot
│   ├── linear_probe.py     CLIP + Linear head
│   └── metrics.py          평가 지표
├── splits/                 splits.json, crop_splits.json (gitignored)
├── data/                   trash-data/, crops/                  (gitignored)
├── outputs/                e2e_predictions.jsonl, crops_e2e/    (gitignored)
├── results/                exp1/, exp2/, dino_eval/             (gitignored)
└── requirements.txt
```

## 카테고리 / 프롬프트 설정

**`config.py`는 `tools/sync_label_mapping.py`가 자동 생성합니다. 직접 편집하지 마십시오.**

| 변경하고 싶은 것 | 수정 위치 | 다음 명령 |
|---|---|---|
| 활성 main / sub 변경 | `src/label_mapping.py` (`ACTIVE_SUBS`, `KEEP_MAINS`, `KOR_TO_EN`) | `python tools/sync_label_mapping.py` |
| dino_prompt 영어 alias | `src/label_mapping.py` (`KOR_TO_EN[main]`) | sync 재실행 |
| sub_category CLIP description | `config.py` 의 `_ALL_CATEGORIES` 안 description 직접 편집 | sync 재실행 (보존됨) |

`config.py` 출력은 3개 dict:
- `_ALL_CATEGORIES` — 96 카테고리 universe (description 저장)
- `ACTIVE_MAIN`     — 31 활성 main (= `KEEP_MAINS`)
- `CATEGORY_CONFIG` — 활성 31 main / 54 sub, dino_prompt = `KOR_TO_EN` aliases

> sub_category 키 정규화: 공백 제거, `cm` → `㎝`. 예: `"소파_ 1인용"` → `"소파_1인용"`.

---

## 실행 순서

데이터(`data/trash-data/csv/`, `data/trash-data/image/`)가 준비된 상태에서:

### Step 0 — config 생성

```bash
python tools/sync_label_mapping.py
```

### Step 1 — Split 생성

```bash
python 00_make_split.py
python 00_make_split.py --max-per-category 0          # 전체 사용
python 00_make_split.py --no-verify-images            # 검증 skip (느린 디스크)
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data-root` | `data/trash-data` | CSV·이미지 루트 |
| `--out` | `splits/splits.json` | 출력 |
| `--train` / `--val` | `0.7` / `0.15` | 분할 비율 |
| `--seed` | `42` | 랜덤 시드 |
| `--max-per-category` | `100` | sub_category별 최대 샘플 (0=전체) |
| `--no-verify-images` | off | PIL decode 검증 skip (≥30k에서 권장) |

> `splits/splits.json`이 존재하면 경고 후 종료. 재생성 시 파일 삭제.

### Step 2 — DINO Crop 추출

```bash
python 01_extract_crops.py
```

각 이미지에 `CATEGORY_CONFIG[main]["dino_prompt"]`로 GDINO를 실행, 최고 신뢰도 박스를 `data/crops/{split}/{file_name}`에 저장하고 `splits/crop_splits.json`을 출력합니다. 검출 실패 시 원본 이미지가 fallback으로 들어갑니다.

`crop_splits.json` 의 각 record는 다음 필드를 갖습니다 (CLIP 담당 인터페이스):

```json
{
  "image_path": "data/crops/test/abc.jpg",
  "file_name": "abc.jpg",
  "main_category": "소파",
  "sub_category": "소파_1인용",
  "group_id": "abc",
  "dino_meta": {
    "detection_success": true,
    "fallback": false,
    "score": 0.78,
    "box": [x0, y0, x1, y1],
    "label_en": "sofa",
    "image_size": [W, H]
  }
}
```

### Step 3 — DINO BBOX 평가

```bash
python 02_dino_eval.py
python 02_dino_eval.py --split test
```

`splits/crop_splits.json` 의 `dino_meta` 통계:
- 전체/per-class detection rate
- success score 분포 (mean / p50 / p90 / min / max)
- fallback / missing_meta / open_error 카운트

출력: `results/dino_eval/{split}_report.json` + 콘솔 요약.

### Step 4 — CLIP 실험 (CLIP 담당)

```bash
python exp1_zeroshot.py --split test         # CLIP only vs DINO+CLIP
python exp2_train.py                         # Linear Probe 학습
python exp2_evaluate.py --checkpoint runs/exp2/<run_id>/best.pt
```

결과: `results/exp1/`, `results/exp2/`.

### E2E inference (GT 없는 입력)

```bash
python 10_e2e_pipeline.py --splits splits/splits.json --split test --limit 30
python 10_e2e_pipeline.py --images data/trash-data/image --limit 100
```

Stage A (single-prompt 31 클래스 GDINO) + Stage B (예측된 main의 per-class crop). 출력 `outputs/e2e_predictions.jsonl`:

```json
{
  "file_name": "...",
  "image_size": [W, H],
  "stage_a": {
    "pred_main": "소파" | null,
    "label_en": "sofa" | null,
    "score": 0.78 | null,
    "box": [x0,y0,x1,y1] | null,
    "topk": [{"label_en","label_kor","score","box"}, ...]
  },
  "stage_b": {
    "crop_path": "outputs/crops_e2e/<file>" | null,
    "dino_prompt": "...",
    "score": 0.74 | null,
    "box": [x0,y0,x1,y1] | null,
    "fallback": true|false|null,
    "label_en": "..." | null
  }
}
```

`stage_a.pred_main == null` → Stage B skip. `stage_b.fallback == true` → main 예측은 됐으나 per-class detection 실패 (전체 이미지 crop).

---

## 데이터 소스

CSV 위치: `data/trash-data/csv/` (4개 파일, 30,099행).
이미지 위치: `data/trash-data/image/` (.jpg).

| 파일 | 설명 |
|---|---|
| `ipcamp_5868_2020-12-21.csv` | 소파, 텔레비전, 의자 등 |
| `ipcamp_5995_2020-12-21.csv` | 텔레비전, 상, 피아노 등 |
| `ipcamp_5996_2020-12-21.csv` | 프린트기, 조명, 소화기 등 |
| `ipcamp_5997_2020-12-21.csv` | 전동안마의자, 정수기, 유모차 등 |

컬럼: `file_name`, `main_category`, `sub_category`.

---

## 담당 영역

- **GDINO**: `src/dino.py`, `src/label_mapping.py`, `01_extract_crops.py`, `02_dino_eval.py`, `10_e2e_pipeline.py`, `tools/sync_label_mapping.py`
- **CLIP**: `src/clip_zeroshot.py`, `src/linear_probe.py`, `exp1_zeroshot.py`, `exp2_*.py`
- **공유**: `00_make_split.py`, `src/dataset.py`, `src/metrics.py`, `config.py`(자동 생성)
