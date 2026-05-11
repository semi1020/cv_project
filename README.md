# cv_project — 대형폐기물 분류

이미지 1장 → main_category(31종) + sub_category(54종) 자동 분류 → 배출수수료 안내.

## 시스템 개요 (Pipeline v2 — 3-stage open-vocabulary)

```
                  ┌─────────────────────────────────────┐
                  │           E2E inference v2          │
                  │       11_e2e_pipeline_v2.py         │
                  └─────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  [Stage A] CLIP-family    [Stage B] GDINO         [Stage C] CLIP-family
  zero-shot (31 main)      per-class single-prompt zero-shot (per-main subs)
  → main_category          → crop                  → sub_category
  (CLIP / MetaCLIP /       (선택: --skip-stage-b)  (Stage A와 동일 모델
   SigLIP / SigLIP2)                                  재사용)
```

**현재 baseline (test 3927장):** SigLIP2-L-512 zero-shot, Stage A **89.2%** main accuracy.
실험 이력 / 의사결정 / 모델 비교 → [docs/PIPELINE_HISTORY.md](docs/PIPELINE_HISTORY.md).

### 설계 원칙
1. **Open-vocabulary 우선** — 새 클래스 추가는 `src/label_mapping.py` 한 줄 수정 + `tools/sync_label_mapping.py` 실행으로 완료. 재학습 불필요.
2. **GDINO + CLIP 모두 zero-shot 활용** — 둘 다 open-vocab 모델. GDINO는 학습된 task인 phrase grounding (Stage B)으로만 사용.
3. **Linear probe는 보조 도구** — 사이즈/규격 sub처럼 zero-shot 본질적 한계 영역에 한정 검토.

### 레거시 파이프라인
[10_e2e_pipeline.py](10_e2e_pipeline.py)는 GDINO single-prompt를 Stage A로 쓰던 v1 (main 57%). 비교용으로 보존, 신규 작업은 v2 사용.

학습/평가 흐름은 GT main을 가정한 별도 경로(`01_extract_crops.py` → `02_dino_eval.py` → `exp1`/`exp2`)를 사용합니다.

## 프로젝트 구조

```
cv_project/
├── 00_make_split.py            Step 0: splits/splits.json
├── 01_extract_crops.py         Step 1: splits/crop_splits.json + data/crops/
├── 02_dino_eval.py             Step 2: results/dino_eval/ (BBOX 평가)
├── 10_e2e_pipeline.py          [LEGACY] Pipeline v1 (GDINO single-prompt Stage A, main 57%)
├── 11_e2e_pipeline_v2.py       [현재 권장] Pipeline v2 (CLIP-A + GDINO-B + CLIP-C)
├── exp1_zeroshot.py            실험 1: CLIP only vs DINO+CLIP (sub 분류, GT main 가정)
├── exp2_train.py               실험 2: Linear Probe 학습 (sub용)
├── exp2_evaluate.py            실험 2: 평가
├── exp_clip_main.py            실험: open-vocab main 분류 (CLIP/MetaCLIP/SigLIP/SigLIP2 swap)
├── config.py                   자동 생성 — 직접 편집 금지
├── tools/
│   ├── sync_label_mapping.py       src/label_mapping.py → config.py 변환
│   ├── run_experiment.sh           Pipeline v1 자동 실행/평가 (legacy 호환)
│   ├── eval_clip_main.sh           main 분류 평가 (v3 포맷 출력)
│   └── eval_e2e_v2.sh              Pipeline v2 E2E 평가 (main + sub)
├── src/
│   ├── label_mapping.py        KOR_TO_EN, ACTIVE_SUBS, KEEP_MAINS (단일 소스)
│   ├── dataset.py              SampleRecord(+dino_meta), split 유틸
│   ├── dino.py                 Grounding DINO base wrapper
│   ├── clip_zeroshot.py        Open-vocab zero-shot (CLIP/MetaCLIP/SigLIP/SigLIP2)
│   ├── linear_probe.py         CLIP + Linear head (보조 도구)
│   ├── prompt_chunks.py        [DEPRECATED] 청킹 모드 (-3.5pp/2x slow → 폐기, 보존)
│   └── metrics.py              평가 지표
├── docs/
│   └── PIPELINE_HISTORY.md     실험/의사결정/아키텍처 변천 이력
├── splits/                     splits.json (git tracked), crop_splits.json (gitignored)
├── outputs/                    e2e_*.jsonl, crops_e2e/         (gitignored)
├── results/                    exp1/, exp2/, dino_eval/        (gitignored)
├── experiments/                실험 로그/eval                  (gitignored — 산출물)
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

데이터(`/data/trash-data/csv/`, `/data/trash-data/image/`)가 준비된 상태에서:

### Step 0 — config 생성

```bash
python tools/sync_label_mapping.py
```

### Step 1 — Split 생성

> **주의**: `splits/splits.json`은 git으로 관리됩니다.
> 이미 존재하는 경우 `git pull`로 받은 파일을 그대로 사용하십시오.
> GDINO/CLIP 담당자가 동일한 train/val/test 분할을 공유해야 합니다.
>
> 현재 split: train 18,855 / val 4,041 / test 4,044 (합계 26,940, 31 classes, seed=42)

```bash
# 최초 생성 시에만 실행 (splits.json이 없을 때)
python 00_make_split.py
python 00_make_split.py --max-per-category 0          # 전체 사용
python 00_make_split.py --no-verify-images            # 검증 skip (느린 디스크)
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data-root` | `/data/trash-data` | CSV·이미지 루트 |
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

각 이미지에 `CATEGORY_CONFIG[main]["dino_prompt"]`로 GDINO를 실행, 최고 신뢰도 박스를 `/data/trash-data/crops/{split}/{file_name}`에 저장하고 `splits/crop_splits.json`을 출력합니다. 검출 실패 시 원본 이미지가 fallback으로 들어갑니다.

`crop_splits.json` 의 각 record는 다음 필드를 갖습니다 (CLIP 담당 인터페이스):

```json
{
  "image_path": "/data/trash-data/crops/test/abc.jpg",
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
python exp1_zeroshot.py --split test         # Sub: CLIP only vs DINO+CLIP (GT main 가정)
python exp2_train.py                         # Sub: Linear Probe 학습 (보조 도구)
python exp2_evaluate.py --checkpoint runs/exp2/<run_id>/best.pt

# Open-vocab main 분류 모델 비교 (CLIP/MetaCLIP/SigLIP/SigLIP2)
python exp_clip_main.py --split test --model-id google/siglip2-large-patch16-512
bash tools/eval_clip_main.sh clip_main_canonical_<model_short> outputs/<...>.jsonl
```

결과: `results/exp1/`, `results/exp2/`, `outputs/clip_main_*.jsonl`.

### E2E inference (GT 없는 입력)

#### 권장 — Pipeline v2 (CLIP-A → GDINO-B → CLIP-C)

```bash
# 기본 (Stage B 포함, SigLIP2-L-512 baseline)
python 11_e2e_pipeline_v2.py --splits splits/splits.json --split test \
  --model-id google/siglip2-large-patch16-512

# Fast 모드 (Stage B 생략, full image로 Stage C 직행)
python 11_e2e_pipeline_v2.py --splits splits/splits.json --split test \
  --model-id google/siglip2-large-patch16-512 --skip-stage-b

# 모델 swap만으로 다른 백본 비교 — 학습 코드 변경 없음 (open-vocab)
python 11_e2e_pipeline_v2.py --splits splits/splits.json --split test \
  --model-id facebook/metaclip-l14-fullcc2.5b
python 11_e2e_pipeline_v2.py --splits splits/splits.json --split test \
  --model-id openai/clip-vit-large-patch14
```

지원 모델 (HF에서 가중치 자동 다운로드):
- `openai/clip-vit-large-patch14` (CLIP-L, 428M, 224px)
- `facebook/metaclip-l14-fullcc2.5b` (MetaCLIP-L, 428M, 224px)
- `google/siglip-large-patch16-384` (SigLIP-L, 652M, 384px)
- `google/siglip2-large-patch16-512` (**SigLIP2-L-512, 652M, 512px — 현재 baseline**)
- 기타 SigLIP/SigLIP2/EVA-CLIP/OpenCLIP 변형 가능

출력 `outputs/e2e_v2_*.jsonl`:
```json
{
  "file_name": "...",
  "image_size": [W, H],
  "stage_a": {"pred_main": "소파" | null, "score": 0.91, "topk": [...]},
  "stage_b": {"crop_path": "outputs/crops_e2e_v2/<file>" | null,
              "dino_prompt": "...", "score": 0.74, "box": [...], "fallback": false},
  "stage_c": {"pred_sub": "소파_1인용" | null, "score": 0.82, "all_scores": {...}},
  "model_id": "google/siglip2-large-patch16-512",
  "stage_c_model_id": "google/siglip2-large-patch16-512",
  "stage_a_mode": "canonical",
  "skip_stage_b": false
}
```

평가:
```bash
bash tools/eval_e2e_v2.sh <exp_name> outputs/e2e_v2_<...>.jsonl
```

#### 레거시 — Pipeline v1 (GDINO single-prompt)

```bash
python 10_e2e_pipeline.py --splits splits/splits.json --split test --limit 30
```

Stage A를 GDINO single-prompt로 사용한 옛 파이프라인 (main 57%). 비교용으로 보존.

---

## 데이터 소스

CSV 위치: `/data/trash-data/csv/` (4개 파일, 30,099행).
이미지 위치: `/data/trash-data/image/` (.jpg).

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
- **CLIP**: `src/clip_zeroshot.py`, `src/linear_probe.py`, `exp1_zeroshot.py`, `exp2_*.py`, `exp_clip_main.py`
- **Pipeline v2 (공유)**: `11_e2e_pipeline_v2.py`, `tools/eval_e2e_v2.sh`, `tools/eval_clip_main.sh` — Stage A/C는 CLIP 담당, Stage B는 GDINO 담당
- **공유 인프라**: `00_make_split.py`, `src/dataset.py`, `src/metrics.py`, `config.py`(자동 생성), `docs/PIPELINE_HISTORY.md`
