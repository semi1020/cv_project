# Pipeline 변동 이력 (Pipeline History)

생활폐기물 분류 파이프라인의 실험 / 의사결정 / 아키텍처 변천 기록.

> **마지막 업데이트:** 2026-05-11
> **현재 아키텍처 (Pipeline v2):** CLIP zero-shot (Stage A) + GDINO per-class (Stage B, 옵션) + CLIP zero-shot / linear probe (Stage C)
> **현재 baseline:** main 89.2% / sub-given-main 83.1% / **E2E 74.1%** ([실측 e2e_v2_siglip2_probe](../experiments/e2e_v2_siglip2_probe_eval.txt)) — SigLIP2-L-512 no_b + linear probe

## 설계 원칙 (Design Principles)

본 프로젝트는 **open-vocabulary** 설계를 핵심 원칙으로 합니다:

1. **새 클래스 추가는 [src/label_mapping.py](../src/label_mapping.py) 한 줄 수정 + `tools/sync_label_mapping.py` 실행으로 완료** — 재학습 불필요
2. **GDINO + CLIP 모두 zero-shot 활용** — 둘 다 open-vocab 모델
3. **Linear probe는 보조 도구** — 시각만으로 결정 불가능한 사이즈/규격 sub 한정. main 분류엔 미사용
4. **연구 이력 보존** — 폐기된 시도(청킹, alias 확장)도 회피 사유와 함께 기록

성능과 설계 원칙이 충돌할 때, 단순 정확도가 아니라 **운영 비용 (재학습 빈도, 클래스 추가 속도)** 까지 고려하여 trade-off를 결정합니다.

---

## 0. 개요

생활폐기물 이미지 → 31개 main_category → 54개 sub_category 분류.
Test split: 3,927장 ([splits/splits.json](../splits/splits.json) test).

### Pipeline 구조 변천

| 시점 | Stage A (main 분류) | Stage B (검출/crop) | Stage C (sub 분류) |
|---|---|---|---|
| ~v3 | GDINO 단일 프롬프트 (31 classes) | GDINO per-class | CLIP zero-shot |
| v6 (시도/폐기) | GDINO 청킹 (3 그룹) | 동일 | 동일 |
| v7 (시도/폐기) | GDINO + 확장 alias | 동일 | 동일 |
| **현재 (Pivot)** | **CLIP zero-shot** | GDINO per-class (옵션) | CLIP zero-shot |

---

## 1. 실험 결과 요약

| 실험 | 날짜 | Stage A 방식 | main_acc | main_fail | det_acc | 시간 | 비고 |
|---|---|---|---|---|---|---|---|
| v4a | 2026-05-09 | GDINO single (alias v4a) | 46.4% | 0.6% | 46.7% | 4703s | 흡수 confusion 큼 |
| v4c | 2026-05-09 | GDINO single (alias v4c) | 55.3% | 3.2% | 57.1% | 4449s | |
| **v3_baseline** | 2026-05-10 | GDINO single (alias v3) | **57.0%** | 2.9% | 58.7% | 4643s | **이전 best** |
| v6_chunked_3g | 2026-05-10 | GDINO 3-group chunked | 53.5% | 1.1% | 54.0% | 10042s | -3.5pp, 2.2x slow → **폐기** |
| v7_alias | 2026-05-10 | GDINO single + PROMPT_BOOST 확장 | 47.0% | 5.2% | 49.5% | 4531s | -10pp 폭락 → **롤백** |
| **clip_main_canonical** | 2026-05-10 | **CLIP zero-shot canonical** | **79.5%** | **0.0%** | **79.5%** | ~30분 | **+22.5pp, 신규 baseline** |
| clip_main_multi | 2026-05-10 | CLIP zero-shot multi-alias avg | 79.5% | 0.0% | 79.5% | ~30분 | total 동일, 분포 재배치 — § 2.4 참고 |
| **e2e_v2_no_b** | 2026-05-10 | **CLIP-A canon + B skip + CLIP-C** | main 79.5% / sub-given-main **61.9%** / **E2E 49.2%** | 0% | - | ~30분 | Pipeline v2 (Stage B 생략 변형) |
| **e2e_v2_full** | 2026-05-10 | **CLIP-A canon + B GDINO + CLIP-C** | main 79.5% / sub-given-main **62.1%** / **E2E 49.4%** | 0% | - | ~80분 | Pipeline v2 baseline (CLIP-L, OpenAI) |
| clip_main_metaclip_L | 2026-05-11 | **MetaCLIP-L** zero-shot canonical | **85.8%** | 0.0% | 85.8% | ~50분 | +6.3pp vs CLIP-L, 동일 크기(428M) |
| **clip_main_siglip_L** | 2026-05-11 | **SigLIP-L** zero-shot canonical | **86.6%** | **0.0%** | **86.6%** | ~50분 | **+7.1pp, 신규 Stage A baseline** |
| **e2e_v2_siglip_no_b** | 2026-05-11 | **SigLIP-A canon + B skip + SigLIP-C** | main 86.6% / sub-given-main **74.6%** / **E2E 64.6%** | 0% | - | ~30분 | Pipeline v2 baseline (SigLIP-L). +15.2pp vs CLIP-L |
| **clip_main_siglip2_L_512** | 2026-05-11 | **SigLIP2-L-512** zero-shot canonical | **89.2%** (3502) | **0.0%** | **89.2%** | ~29분 | **+2.6pp vs SigLIP-L, 더 빠름. 신규 Stage A baseline** |
| direct_sub_siglip2 | 2026-05-11 | SigLIP2-L-512 **직접 54-way sub 분류** (Stage 없음) | implied 82.0% | 0.0% | - | ~35분 | E2E 62.52% — Pipeline보다 열등. § 2.9 |
| **e2e_v2_siglip2_no_b** | 2026-05-11 | **SigLIP2-A canon + B skip + SigLIP2-C** | main **89.2%** / sub-given-main **77.25%** / **E2E 68.91%** | 0% | - | ~37분 | **공식 baseline (probe 이전). § 2.10** |
| **e2e_v2_siglip2_probe** | 2026-05-11 | **SigLIP2-A canon + B skip + linear probe-C** | main **89.2%** / sub-given-main **83.1%** / **E2E 74.1%** | 0% | - | ~35분 | **신규 공식 baseline. +5.19pp vs no_b. § 2.11** |

데이터 원본: [experiments/summary.csv](../experiments/summary.csv), [clip_main_canonical_eval.txt](../experiments/clip_main_canonical_eval.txt), [clip_main_multi_eval.txt](../experiments/clip_main_multi_eval.txt), [e2e_v2_no_b_eval.txt](../experiments/e2e_v2_no_b_eval.txt), [e2e_v2_full_eval.txt](../experiments/e2e_v2_full_eval.txt), [clip_main_canonical_siglip-large-patch16-384_eval.txt](../experiments/clip_main_canonical_siglip-large-patch16-384_eval.txt), [clip_main_canonical_metaclip-l14-fullcc2.5b_eval.txt](../experiments/clip_main_canonical_metaclip-l14-fullcc2.5b_eval.txt), [e2e_v2_siglip-large-patch16-384_canonical_no_b_eval.txt](../experiments/e2e_v2_siglip-large-patch16-384_canonical_no_b_eval.txt), [clip_main_canonical_siglip2-large-patch16-512_eval.txt](../experiments/clip_main_canonical_siglip2-large-patch16-512_eval.txt)

---

## 2. 핵심 의사결정 이력

### 2.1 청킹 폐기 (2026-05-10, v6 → v3 복귀)

**가설:** 31개 클래스를 그룹별로 분리해 GDINO를 그룹 수만큼 실행 → 그룹간 score max 선택.

**결과:** -3.5pp, 시간 2.2배. 다음 클래스에서 큰 손실:
- 청소기 73.6% → 42.0% (-31.6pp)
- 선풍기 64.9% → 42.1% (-22.8pp)
- 형광등기구 42.2% → 22.2% (-20.0pp)

**원인 진단:**
1. GDINO score는 prompt context-dependent → 그룹 간 score 비교 비공정
2. 같은 그룹에 묶인 혼동 페어(의자/소파/상)는 confusion 폭증
3. 그룹 수만큼 GDINO forward → 시간 비용

**조치:** `--chunked` 옵션 [10_e2e_pipeline.py](../10_e2e_pipeline.py)에서 제거. [src/prompt_chunks.py](../src/prompt_chunks.py)는 deprecation 헤더 추가 후 보관.

### 2.2 PROMPT_BOOST 확장 실패 (2026-05-10, v7 롤백)

**가설:** v3에서 0% 정확도 7개 클래스(화장대/오락기/의료기/식탁/책상/컴퓨터/TV장식장)의 alias를 확장해서 정확도 끌어올림. 'vanity table' 추가 등.

**결과:** -10pp 폭락. 약한 클래스는 개선됐으나 강한 클래스가 흡수당함:
- **상: 62.6% → 0.6%** (-62pp, 'vanity table'이 'low table' 흡수)
- 텔레비전: 64.7% → 46.3% ('monitor'/'computer desk'에 흡수)
- 공기청정기: 61.6% → 40.6%
- 의자: 67.6% → 53.4% ('chairs' 토큰 누출, NO_DET 64건 발생)

**원인 진단:**
1. **Prompt engineering = zero-sum**. 약한 클래스 alias 추가 = 강한 클래스 흡수.
2. **다중 단어 alias는 토큰 단위 누출**. 'tall dining table with chairs' → 'chairs' 토큰이 의자 detection 깨뜨림.
3. **비대칭 클래스 분포 효과**. 약한 클래스(15~50개) +pp는 작고, 강한 클래스(174~960개) -pp는 큼. 수학적으로 net negative.

**조치:** [src/label_mapping.py](../src/label_mapping.py)를 v3로 롤백 (`git checkout`).

### 2.3 아키텍처 전환: GDINO → CLIP for Stage A (2026-05-10) ⭐

**진단 (CV/GDINO 관점):**
- GroundingDINO는 **phrase grounding** task로 학습됨: "주어진 짧은 phrase 하나에 박스 찾기"
- 우리는 GDINO를 **31-way classifier**로 사용 중: 모든 클래스를 한 prompt에 나열, max-score 선택
- 이는 **tool-task mismatch**. 다음 증거들이 가리킴:
  - GDINO 출력에 phrase boundary 깨진 garbled 라벨: 'low table table', 'vanity table', 'chair office chair armchair'
  - Token leakage: 'chairs' → 의자 detection 깨뜨림
  - Prompt engineering의 모든 시도가 zero-sum
- CLIP은 **image-text matching**으로 학습됨 = 본질적으로 분류기

**가설 검증:** 동일 alias로 CLIP zero-shot main 분류 측정.

**결과:**
- v3 GDINO: 57.0% main_acc
- CLIP canonical: **79.5% main_acc** (+22.5pp, prompt 수정 없이)
- main_fail: 2.9% → 0.0% (CLIP은 항상 prediction을 냄)
- 시간: 75분 → 30분

**조치:** Stage A 후속 실험은 모두 CLIP 기반으로 진행. GDINO는 Stage B (per-class single-prompt detection — 학습된 용도)로만 사용.

### 2.4 CLIP multi-alias 실험: zero-sum 재분포 발견 (2026-05-10)

**가설:** canonical(클래스당 1 alias) 대비 multi-alias 평균 임베딩이 약한 클래스 회복.

**결과:** **total 정확도 정확히 동일 (79.5%, 둘 다 3121/3927)**, 클래스별 분포만 재배치.

| 변화 방향 | 클래스 (델타) | 절대수 변화 |
|---|---|---|
| 큰 폭 개선 | 컴퓨터 +52.8pp, 의료기 +42.9pp, 화장대 +16.7pp, 형광등 +15.6pp, 텔레비전 +9.1pp, 에어컨 +11.5pp, 상 +11.5pp | +130건 |
| 큰 폭 손실 | 의자 -8.9pp (−85건), 거울 -24.2pp, 세탁기 -17.7pp, 냉장고 -8.1pp, 실내조명 -7.4pp | -130건 |

**원인 진단:** CLIP은 클래스별 임베딩이 독립이지만 **임베딩 공간의 의미적 영역이 한정**됨. 한 클래스가 alias를 늘려 영역을 확장하면 인접 클래스가 줄어듬 (semantic-level zero-sum, GDINO의 token-level zero-sum보다는 약함). 거울은 alias 1개뿐인데도 -24.2pp 손실 — 화장대의 'vanity with mirror'/'dressing table with mirror'가 'mirror' 토큰을 차지함.

**조치:** Pipeline v2 통합엔 **canonical 채택** (의자 -85건 손실 회피). 약한 클래스(의료기, 컴퓨터)는 CLIP 모델 업그레이드(SigLIP/EVA-CLIP) 또는 사이즈 sub linear probe(보조)로 해결. 선택적 하이브리드 alias는 다시 prompt engineering 굴레라 회피.

### 2.5 Pipeline v2 baseline 확립 + Stage B 가치 측정 (2026-05-10)

**실험:** 신규 [11_e2e_pipeline_v2.py](../11_e2e_pipeline_v2.py)로 main+sub E2E 평가 2회 (B 포함/생략).

**결과:**
| 변형 | main_acc | sub-acc (given main) | E2E | 시간 |
|---|---|---|---|---|
| v2_no_b (Stage B 생략) | 79.5% | 61.9% | 49.2% | ~30분 |
| v2_full (Stage B 포함) | 79.5% | 62.1% | 49.4% | ~80분 |

**핵심 관찰:**
- Stage B 가치 = +0.2pp sub-acc (+6/3927건). **단순 분류 ROI는 미미.**
- Per-class zero-sum: 청소기 +15건, 텔레비전 -16건. crop이 핵심 영역 잘라낼 때 손해.
- exp1 (DINO crop +0.1pp F1)과 일관 — 두 번 확인.
- Sub 약점은 **사이즈/규격 confusion**: 공기청정기 1m미만→이상 222건, 의자 안락→보조 97건 등.

**의사결정:** Stage B **유지**. 근거:
1. **설계 일관성** (open-vocab GDINO 컴포넌트 보존)
2. **연구 이력 보존** — GDINO 사용 흔적은 향후 better detector 비교 baseline
3. **분류 외 활용 가능성** (localization, 시각화, 데이터 수집 보조)
4. 정확도 +0.2pp는 작지만 zero가 아님

단, `--skip-stage-b` 옵션 유지하여 fast inference 모드 가능.

### 2.6 CLIP 모델 업그레이드 — open-vocab 핵심 lever 검증 (2026-05-11)

**가설:** CLIP 백본을 더 강력한 open-vocab 모델(MetaCLIP, SigLIP)로 swap하면 prompt 수정 없이도 Stage A 정확도가 크게 오를 것.

**구현:** [src/clip_zeroshot.py](../src/clip_zeroshot.py)를 `AutoProcessor`/`AutoModel`로 일반화 + `--model-id` 인자. SigLIP은 sigmoid scoring 자동 분기. **Open-vocab 워크플로우 영향 없음** (모델 swap만, prompt/학습 동일).

**결과 (canonical 모드, 동일 prompt):**

| 모델 | params | main acc | Δ vs CLIP-L | 절대 회복 |
|---|---|---|---|---|
| CLIP-L (OpenAI, baseline) | 428M | 79.5% | - | - |
| MetaCLIP-L | 428M | **85.8%** | **+6.3pp** | +248건 |
| **SigLIP-L** | 652M | **86.6%** | **+7.1pp** | **+279건** |

**Per-class 핵심 패턴:**

SigLIP가 압도적으로 회복한 클래스 (≥+15pp vs CLIP-L):
- 식탁 38.6% → **94.7%** (+56pp), 컴퓨터 22.2% → 75.0% (+52pp)
- 세면대 61.9% → 100%, 책상 16.7% → 50%, 거울 60.6% → 90.9%
- 화장대 61.1% → 83.3%, 전자레인지 66.7% → 87.5%
- TV장식장 74.1% → 92.6%, 냉장고 76.8% → 94.9%

MetaCLIP만의 강점 (SigLIP보다 우위):
- 텔레비전 84.7% (SigLIP 77.5%, CLIP 62.6%)
- 컴퓨터 80.6% (SigLIP 75.0%)
- 상 94.3% (SigLIP 75.3% — SigLIP에선 회귀)
- 오락기 11.1% (유일하게 0% 탈출)

전 모델 0% 잔존: **의료기** — canonical='wheelchair'가 실제 이미지(walker/mobility aid)와 매칭 안 됨. 모델 swap으론 해결 안 되는 alias 문제.

**이론적 ensemble 천장 (per-class best routing):** ≈ 89.2% — SigLIP 단독 대비 +2.6pp 가능하나 라우팅 복잡도 대비 효과 적음.

**의사결정:** 
1. **SigLIP-L을 신규 Stage A baseline으로 채택**. CLIP-L은 비교 baseline으로 보존.
2. Pipeline v2 E2E 재실행 시 SigLIP-L 사용 (Stage A + C 동일).
3. 의료기/오락기 잔존 약점은 별도 트랙 (alias 미세 조정 또는 사이즈 sub probe 시 함께 검토).

### 2.7 Pipeline v2 SigLIP-L E2E — 사이즈 sub 일부 cracked (2026-05-11)

**실험:** [11_e2e_pipeline_v2.py](../11_e2e_pipeline_v2.py) `--model-id google/siglip-large-patch16-384 --skip-stage-b`.

**핵심 발견:** Stage A +7.1pp가 sub까지 +12.5pp (예상보다 큼). **SigLIP의 384px 고해상도가 사이즈 시각 단서까지 보존**한 것으로 보임.

| 지표 | CLIP-L (v2_full) | **SigLIP-L (no_b)** | Δ |
|---|---|---|---|
| main | 79.5% | **86.6%** | +7.1pp |
| sub-given-main | 62.1% | **74.6%** | **+12.5pp** |
| **E2E** | 49.4% | **64.6%** | **+15.2pp / +598건** |

**Sub 회복 큰 클래스 (이전 사이즈 confusion 영역):**
| 클래스 | CLIP-L sub | SigLIP-L sub | Δ | 비고 |
|---|---|---|---|---|
| 공기청정기 | 24.7% | **79.9%** | **+55.2pp** | 1m미만→이상 222 confusion이 거의 해소됨 |
| 소파 | 21.2% | 64.4% | +43.2pp | 5-way 인용수/카우치/스툴 구분 회복 |
| 청소기 | 61.4% | 88.3% | +26.9pp | 가정용/업소용 |
| 진열장 | 51.0% | 75.0% | +24.0pp | 가로 90㎝ |
| 냉장고 | 42.1% | 66.0% | +23.9pp | 용량 (300/500ℓ) |
| 상 | 38.5% | 55.0% | +16.5pp | 4인용 |
| 텔레비전 | 67.4% | 80.8% | +13.4pp | 30인치 |

**Sub 회귀 (조사 필요, 절대수 작음):**
| 클래스 | CLIP-L sub | SigLIP-L sub | Δ | 추정 원인 |
|---|---|---|---|---|
| 거울 | 85.0% | 50.0% | **-35.0pp** | 1㎡ 미만/이상 — sigmoid scoring이 wrong class에 더 confident? |
| 실내조명 | 53.7% | 36.4% | -17.3pp | 장식용/일반 (의미적 sub) |
| 에어컨 | 32.3% | 26.7% | -5.6pp | 면적 3-way |
| 컴퓨터 | 100% | 96.3% | -3.7pp | 표본 수 작음 |

**잔존 sub 약점 (사이즈/면적 구분 본질):**
- 에어컨 26.7%, 형광등 31.6%, 실내조명 36.4%, 거울 50%, 상 55%, 소파 64.4%
- → § 5 "사이즈 sub 보조 linear probe" 트랙의 우선 대상

**의사결정:** **SigLIP-L Pipeline v2를 신규 정식 baseline으로 채택**. 다음 lever 후보:
1. 더 큰 / 고해상도 SigLIP (SigLIP-SO400M, SigLIP2-L-512) — 사이즈 단서 추가 보존 가능성
2. 회귀 클래스 (거울/실내조명) 조사
3. 사이즈 sub 보조 linear probe (잔존 약점)

### 2.8 SigLIP2-L-512 — Stage A 89.2% 도달, 이론적 천장 근접 (2026-05-11)

**실험:** [exp_clip_main.py](../exp_clip_main.py) `--model-id google/siglip2-large-patch16-512` (652M, 512px 입력).

**결과:** Stage A **89.2%** (3502/3927) — § 2.6에서 계산된 per-class ensemble 천장(89.2%)에 정확히 도달.

| 모델 | params | input | main acc | Δ vs CLIP-L | 시간/img |
|---|---|---|---|---|---|
| CLIP-L | 428M | 224 | 79.5% | - | ~0.71s |
| SigLIP-L | 652M | 384 | 86.6% | +7.1pp | ~0.76s |
| **SigLIP2-L-512** | 652M | **512** | **89.2%** | **+9.7pp** | **0.442s** ← 더 빠름 |

**SigLIP2-L-512 vs SigLIP-L (384) 변화:**

큰 폭 회복 (>+5pp):
- 상 75.3% → **90.8%** (+15.5pp) ← SigLIP-L 회귀 회복
- 텔레비전 77.5% → **93.0%** (+15.5pp)
- 전자레인지 87.5% → 95.8% (+8.3pp), 에어컨 74.7% → 81.2% (+6.5pp)
- 화장대 +5.6pp, 오락기 +3.7pp (작지만 0% 탈출)

큰 폭 회귀 (조사 필요):
- **진열장 47.3% → 24.7% (-22.6pp)** ← 가장 큰 단일 손실
  - confusion: 진열장 → TV장식장 39, 진열장 → 책상 20
  - 추정: 512px 고해상도가 진열장의 도어/선반 패턴을 TV stand로 인식
- TV장식장 -9.3pp, 식탁 -8.7pp, 책상 -8.3pp

**잔존 0% 클래스 (모델 swap으로 해결 안 됨):**
- 의료기 0% (전 모델 동일) — canonical 'wheelchair' 매칭 실패. **별도 alias 트랙 필요**.

**의사결정:** SigLIP2-L-512 채택. 다음 단계:
1. Pipeline v2 E2E 재측정 (E2E 70%+ 가능성)
2. 진열장 회귀 조사 (또는 SigLIP-L과의 라우팅)
3. 의료기 alias 단독 실험

### 2.9 Direct 54-way 분류 실험 — Pipeline 분해의 가치 역으로 확인 (2026-05-11)

**가설:** Stage A/B를 거치지 않고 SigLIP2-L-512로 54개 sub_category를 한 번에 분류하면 어떨까? Main 오분류 누적 오류 없이 바로 sub를 맞힐 수 있을지 검증.

**실험:** [exp_clip_direct_sub.py](../exp_clip_direct_sub.py) — `CATEGORY_CONFIG`의 54개 sub 프롬프트를 flat dict으로 합쳐 단일 54-way zero-shot 분류.

**결과:**

| 지표 | Direct 54-way | Pipeline v2 SigLIP-L (no_b) | 비고 |
|---|---|---|---|
| implied main_acc | 82.0% | 86.6% (Stage A) | -4.6pp |
| **sub_acc (E2E)** | **62.52%** | **64.6%** | **-2.1pp** |

→ **Pipeline이 더 좋음.** Direct 54-way는 열등하다.

**원인 1: Cross-main confusion 차단막 상실**

Pipeline에서 Stage A가 main을 먼저 확정하면 Stage C는 같은 main 내에서만 경쟁. 직접 54-way에서는 이 차단막이 없어 주요 오류 폭발:

| 오류 패턴 | 건수 |
|---|---|
| 텔레비전 → TV장식장 | 119건 |
| 의자 → 소파 | 94건 |
| 공기청정기 → 에어컨 | 82건 |
| 상 → TV장식장 | 63건 |

**원인 2: 0% "중간 구간" 클래스 6개 완전 소멸**

연속 척도의 경계 구간이 양 극단에 흡수됨:
- 에어컨_1.0㎡미만 (90건) → 0.5㎡미만 64건 + 1.0㎡이상 25건
- 공기청정기_높이1m이상 (63건) → 1m미만 45건 + 에어컨 17건
- 형광등_길이1m이상 (15건) → 길이1m미만 15건 (전부)
- 소파_2인용 (33건), 소파_카우치 (18건), 진열장_가로90㎝미만 (51건)

**원인 3: 식탁 catastrophe (main 10.5%)** — pipeline Stage A에서 높은 정확도를 보이는 식탁이 직접 54-way에서 상 sub 프롬프트에 흡수됨. 51/57건이 상_4인용미만/이상으로 오분류.

**예외적으로 Direct ≥ Pipeline인 클래스:** 에어컨(+15.1pp), 실내조명(+12.4pp), 거울(+25.8pp) — 이는 직접 분류의 장점이 아니라 SigLIP2-L-512 > SigLIP-L 모델 차이.

**의사결정:** 직접 54-way 방식 기각. Pipeline 분해(Stage A → Stage C)의 가치가 실험으로 확인됨. `exp_clip_direct_sub.py`는 비교 기준용으로 보관.

### 2.10 Pipeline v2 SigLIP2-L-512 E2E — probe 도입 전 baseline (2026-05-11)

**실험:** [11_e2e_pipeline_v2.py](../11_e2e_pipeline_v2.py) `--model-id google/siglip2-large-patch16-512 --skip-stage-b` (3,927장, 2202.7s)

**결과:**

| 지표 | CLIP-L v2_full | SigLIP-L no_b | **SigLIP2-L-512 no_b** | Δ vs SigLIP-L |
|---|---|---|---|---|
| main_acc | 79.5% | 86.6% | **89.20%** | +2.6pp |
| sub-given-main | 62.1% | 74.6% | **77.25%** | +2.65pp |
| **E2E** | 49.4% | 64.6% | **68.91%** | **+4.3pp** |

**Sub-given-main per class 변화 (SigLIP-L → SigLIP2-L-512):**

| 클래스 | SigLIP-L | SigLIP2-L-512 | Δ |
|---|---|---|---|
| 거울 | 50.0% | **79.3%** | **+29.3pp** |
| 실내조명 | 36.4% | **60.3%** | **+23.9pp** |
| 에어컨 | 26.7% | **43.2%** | **+16.5pp** |
| 텔레비전 | 80.8% | 83.5% | +2.7pp |
| 공기청정기 | 79.9% | 80.3% | +0.4pp |
| 소파 | 64.4% | 56.4% | **-8.0pp** ▼ |
| 냉장고 | 66.0% | 63.0% | -3.0pp |
| 청소기 | 88.3% | 87.0% | -1.3pp |

소파 -8.0pp 회귀: 512px 고해상도가 소파 sub 구분에 역효과 추정.

**잔존 약점 (sub-given-main 기준):**
- 에어컨 43.2% (3-way 크기, 1.0㎡미만이 0.5㎡미만에 55건 흡수)
- 소파 56.4% (5-way 인용수)
- 진열장 34.8% (main acc 24.7%로 이미 낮아 impact 제한적)
- 형광등 27.8% (main acc 40.0%)

**의사결정:** SigLIP2-L-512 Pipeline v2 (no_b)를 probe 실험의 기준 baseline으로 채택. 다음 lever:
1. **사이즈 sub 보조 linear probe** (§ 2.11 — 완료)
2. 의료기/오락기 alias 조정 (소규모, 46건)
3. SigLIP2-SO400M 시도 (수확 체감, +1~2pp 예상)

### 2.11 Linear Probe Stage C — E2E 74.1% 달성 (2026-05-11)

**실험:** per-main `nn.Linear(1024, n_subs)` probe 17개를 SigLIP2-L-512 임베딩 위에 학습. Stage C에서 `pred_main`이 probe 대상이면 probe 사용, 아니면 CLIP zero-shot으로 fallback.

**파이프라인:**
1. [exp_probe_embed.py](../exp_probe_embed.py) — train/val/test 임베딩 추출 (`runs/probes/embeds_*.pt`)
2. [exp_probe_train.py](../exp_probe_train.py) — per-main probe 학습 300 epoch (Adam lr=1e-3, wd=1e-4, full-batch, val-acc checkpoint)
3. [11_e2e_pipeline_v2.py](../11_e2e_pipeline_v2.py) `--probe-dir runs/probes` — E2E 통합

**결과:**

| 지표 | no_b (CLIP zero-shot C) | **no_b + probe** | Δ |
|---|---|---|---|
| main_acc | 89.2% | 89.2% | — |
| sub-given-main | 77.25% | **83.1%** | **+5.85pp** |
| **E2E** | 68.91% | **74.1%** | **+5.19pp** |

**클래스별 probe 효과 (CLIP zero-shot baseline → probe):**

큰 폭 개선:
| 클래스 | CLIP | Probe | Δ |
|---|---|---|---|
| TV장식장(거실장) | 51.1% | **75.6%** | +24.5pp |
| 진열장(장식장,책장,찬장) | 34.8% | **56.5%** | +21.7pp |
| 실내조명등기구 | 60.3% | **74.3%** | +17.0pp |
| 상 | 61.4% | **77.8%** | +16.4pp |
| 소파 | 56.4% | **72.4%** | +15.8pp |
| 에어컨및온풍기 | 43.2% | **57.7%** | +14.5pp |
| 컴퓨터 | 85.7% | **100.0%** | +14.3pp |
| 의자 | 76.4% | **84.6%** | +8.2pp |

회귀 (probe가 과적합 또는 CLIP이 이미 강했던 클래스):
| 클래스 | CLIP | Probe | Δ |
|---|---|---|---|
| 자전거 | 98.1% | 90.7% | -7.4pp |
| 소화기 | 71.9% | 64.9% | -7.0pp |
| 가방 | 100.0% | 97.8% | -2.2pp |

**잔존 sub 약점 (probe 후):**
- 형광등기구 27.8% — CLIP baseline과 동일. main acc도 40%로 낮아 데이터 부족
- 냉장고 63.0% — probe 개선 없음 (CLIP baseline 동일)
- 에어컨 57.7% — 개선되었으나 여전히 낮음. top confusion: 0.5㎡미만→1.0㎡미만 32건, 1.0㎡이상→1.0㎡미만 25건
- 소화기 64.9% — probe 회귀. 샘플 수 적고 probe 과적합

**Top sub confusion (with probe):**
- 의자_보조,간이 → 의자_편의용: 72건 (시각적 구분 어려움)
- 텔레비전_30인치미만 → 30인치이상: 55건 (참조물 없이 크기 판단 불가)
- 공기청정기_높이1m이상 → 높이1m미만: 49건 (probe 후에도 잔존)

**의사결정:** SigLIP2-L-512 + linear probe를 **신규 공식 baseline으로 채택**. 다음 lever:
1. 소화기/자전거 probe 회귀 조사 — CLIP만 사용하는 selective fallback 검토
2. 에어컨/형광등 추가 개선 (데이터 증강, hard negative 추가)
3. 의료기/오락기 alias 조정 (main acc 0~7%)

---

## 3. 신규 아키텍처 (Pipeline v2)

### 3.1 구조

```
[Stage A] CLIP zero-shot (or linear probe) on full image
    ↓ main_category top-1 (+ top-K candidates)
[Stage B] GDINO single-class prompt for chosen main
    ↓ crop (or skip if marginal)
[Stage C] CLIP zero-shot (or linear probe) on crop
    ↓ sub_category
```

### 3.2 도구별 역할 매트릭스

| 도구 | 학습 task | 사용처 | 회피처 |
|---|---|---|---|
| CLIP | Image-text matching | 분류 (Stage A, C) | 검출 |
| GDINO | Phrase grounding | 단일 클래스 검출 (Stage B) | 다중 클래스 분류 (Stage A 단일 프롬프트) |
| Linear probe | Supervised on CLIP embeddings | 사이즈/규격 sub 분류 (B1 트랙) | - |

### 3.3 클린 디자인 원칙

1. **GDINO에 31-way prompt 넣지 않음**. token leakage 원천 차단.
2. **CLIP은 클래스별 독립 임베딩** → prompt 추가가 zero-sum 아님 (안전하게 alias 추가 가능).
3. **Stage 간 책임 분리** — 분류는 분류기에, 검출은 검출기에.

---

## 4. 약점 클래스 분석 (CLIP canonical 기준)

### 4.1 약점 클래스 (Pipeline v2 baseline 기준)

#### Main 약점 (CLIP zero-shot 한계)

| 클래스 | main acc | 원인 추정 | 대응 (open-vocab 우선) |
|---|---|---|---|
| 의료기 | 0.0% | canonical='wheelchair'만 | CLIP 모델 업그레이드 (SigLIP) |
| 오락기 | 3.7% | canonical='arcade cabinet' | CLIP 업그레이드 |
| 책상 | 16.7% | 'desk' 단독, 상에 흡수 | CLIP 업그레이드 |
| 컴퓨터 | 22.2% | 'computer' 단독, monitor 미노출 | CLIP 업그레이드 |
| 식탁 | 38.6% | 시각적으로 '상' (low table)와 모호 | CLIP 업그레이드 (시각 표현력) |
| 형광등기구 | 42.2% | 의미적으로 실내조명등의 sibling | CLIP 업그레이드 |

#### Sub 약점 (사이즈/규격 confusion — 본질적 한계)

[v2_full top sub confusions](../experiments/e2e_v2_full_eval.txt) 기준:

| 클래스 | sub acc | 핵심 confusion | Top 오류 |
|---|---|---|---|
| 소파 | **23.3%** | 1인용/2인용/3인용/카우치/스툴 (5-way) | 3인용→2인용 81건 |
| 공기청정기 | **24.1%** | 높이 1m 미만/이상 | 1m미만→이상 **222건** ← 단일 최대 오류 |
| 상 | 35.0% | 4인용 미만/이상 | 4인용미만→이상 85건 |
| 에어컨 | 36.1% | 1.0㎡ 이상/미만/0.5㎡미만 | - |
| 냉장고 | 40.8% | 300ℓ 미만/이상/500ℓ이상 | - |
| 형광등 | 42.1% | 길이 1m 미만/이상 | - |
| 진열장 | 45.1% | 가로 90㎝ 미만/이상 | - |
| 실내조명 | 51.2% | 장식용/일반 (의미적) | 장식용→일반 46건 |
| 청소기 | 54.0% | 가정용/업소용 | 가정용→업소용 65건 |

→ CLIP zero-shot은 **이미지에서 절대 사이즈/길이/용량을 측정하지 못함**. CLIP 모델 업그레이드도 본질적 한계. **사이즈 sub만 보조 linear probe 검토** (open-vocab 워크플로우 영향 최소화).

### 4.2 새로운 confusion 패턴 (CLIP)

GDINO에서는 token-level cross-talk 기반 confusion이었으나, CLIP은 **시각적 유사성** 기반:
- 텔레비전 → TV장식장 138회 (TV가 stand 위에 놓인 사진)
- 에어컨 → 공기청정기 76회 (tower-like 형태)
- 의자 → 상 63 (작은 의자가 low table처럼 보임)
- 형광등 → 실내조명등 22 (의미적 sibling)

→ Prompt만으로는 해결 어려움. **Linear probe 또는 supervised classifier 필요**.

---

## 5. 다음 단계 (Roadmap) — Open-vocab 우선

### ✅ Step 1~2 (완료): Pipeline v2 baseline 확립
- CLIP zero-shot Stage A 79.5% (canonical/multi 모두 검증)
- E2E 49.4% 실측 (Stage B 포함 정식 baseline)
- Stage B 가치 +0.2pp 측정 → 유지 결정 (§ 2.5)

### ✅ Step 3 (완료): CLIP 모델 업그레이드 + Pipeline v2 SigLIP 통합
- Stage A: SigLIP-L **86.6%** (+7.1pp), MetaCLIP-L 85.8% (보존)
- Pipeline v2 SigLIP-L E2E: **64.6%** (+15.2pp / +598건). Sub-acc도 +12.5pp.
- 사이즈 sub 일부 cracked (공청기 1m 24.7→79.9pp, 소파 21→64pp 등)
- 상세: § 2.6 / § 2.7

### ✅ Step 3a (완료): SigLIP2-L-512 채택
- Stage A: SigLIP2-L-512 **89.2%** (+2.6pp vs SigLIP-L, +9.7pp vs CLIP-L baseline)
- 이론적 ensemble 천장(89.2%)에 정확히 도달
- 상세: § 2.8

### ✅ Step 3b (완료): SigLIP2-L-512로 Pipeline v2 E2E 재측정
- E2E **68.91%** (main 89.2% × sub-given-main 77.25%)
- SigLIP-L 대비 +4.3pp E2E, CLIP-L 대비 +19.5pp
- 신규 공식 baseline. 상세: § 2.10

### ✅ Step 3b+ (완료, 부산물): Direct 54-way 실험
- 직접 54-way sub 분류 E2E 62.52% — Pipeline보다 열등 (-6.4pp)
- Pipeline 분해의 가치(cross-main confusion 차단)를 역으로 확인
- 상세: § 2.9

### 📋 Step 3c (선택): SigLIP2-SO400M-512 시도
SigLIP2 SO400M 변형이 HF에 있으면 +1~2pp 추가 가능 (수확 체감). 메모리 ~3-4GB 필요 (현재 SigLIP2-L 약 2GB).

### 📋 Step 4 (Step 3 완료 후): GDINO 대안 detector 비교
**연구 이력 보존:** 현재 GDINO Stage B는 그대로 유지하면서 다른 open-vocab detector를 별도 실험으로 비교.

**후보:**
- OWL-ViT v2 (`google/owlv2-large-patch14-ensemble`)
- Grounding DINO 1.5 (newer 버전)
- YOLO-World (real-time open-vocab)

비교 메트릭: per-class detection rate, score distribution, crop quality (Stage C 영향).

### ✅ Step 5 (완료): 사이즈/규격 sub 보조 linear probe
**범위 한정:** main 분류엔 미적용. multi-sub 클래스 17개 대상. 새 main 추가 워크플로우는 영향 없음.

**결과:** E2E **74.1%** (68.91% → +5.19pp). sub-given-main 77.25% → 83.1% (+5.85pp). 상세: § 2.11

**구조:**
- SigLIP2-L-512 임베딩(1024차원) 위에 `nn.Linear(1024, n_subs)` per-main 학습
- Stage A 후 `pred_main ∈ probe_targets`이면 linear probe로 sub 예측, 아니면 CLIP zero-shot 유지
- `--probe-dir` 옵션으로 on/off 전환 (backward compat)

**파일:**
- [exp_probe_embed.py](../exp_probe_embed.py) — 임베딩 추출
- [exp_probe_train.py](../exp_probe_train.py) — per-main probe 학습 + 단독 평가
- [11_e2e_pipeline_v2.py](../11_e2e_pipeline_v2.py) `--probe-dir` — 파이프라인 통합

**제약:** 새 sub 추가 시 retraining 필요 (사이즈 sub 한정이므로 수용 가능한 범위).

---

## 6. 교훈 (Lessons Learned)

1. **도구의 학습 task를 무시하지 말 것**. GDINO는 검출기, CLIP은 분류기. 잘못 쓰면 prompt 튜닝으로는 해결 안 됨.
2. **Prompt engineering은 zero-sum일 수 있음**. 특히 GDINO 단일 프롬프트 내 다중 클래스. 약한 클래스 alias 추가 = 강한 클래스 흡수 위험.
3. **다중 단어 alias의 token leakage**. 'X with Y' 패턴은 Y 토큰이 다른 클래스에 누출.
4. **빠른 가설 검증의 가치**. CLIP zero-shot 30분 실험이 GDINO 75분 실험 5번보다 큰 정보량.
5. **기존 인프라 활용**. exp1, clip_zeroshot.py, linear_probe.py가 이미 있었으나 main 분류에 활용 안 함 → 이게 가장 큰 미스.
6. **설계 원칙은 단순 정확도와 trade-off 결정에 우선**. open-vocab 유지보수성을 위해 +0.2pp 성능을 포기하지 않을 수 있고, 반대로 정확도를 위해 일부 영역(사이즈 sub)에서 retraining을 수용할 수도 있음. 선택은 운영 비용까지 포함한 종합 판단.

---

## 부록 A. 파일 변경 이력

| 날짜 | 파일 | 변경 |
|---|---|---|
| 2026-05-10 | [10_e2e_pipeline.py](../10_e2e_pipeline.py) | `--chunked` flag 및 `_stage_a_chunked` 함수 제거 |
| 2026-05-10 | [src/prompt_chunks.py](../src/prompt_chunks.py) | deprecation 헤더 추가 (보관) |
| 2026-05-10 | [tools/run_experiment.sh](../tools/run_experiment.sh) | chunked 모드 분기 제거 |
| 2026-05-10 | [src/label_mapping.py](../src/label_mapping.py) | v7 alias 확장 시도 → v3로 롤백 |
| 2026-05-10 | [exp_clip_main.py](../exp_clip_main.py) | 신규 — CLIP zero-shot main 분류 (canonical/multi 모드) |
| 2026-05-10 | [tools/eval_clip_main.sh](../tools/eval_clip_main.sh) | 신규 — CLIP main 평가 스크립트 (v3 포맷) |
| 2026-05-10 | [11_e2e_pipeline_v2.py](../11_e2e_pipeline_v2.py) | 신규 — Pipeline v2 (CLIP-A + GDINO-B + CLIP-C) |
| 2026-05-10 | [tools/eval_e2e_v2.sh](../tools/eval_e2e_v2.sh) | 신규 — Pipeline v2 E2E 평가 (main + sub) |
| 2026-05-10 | [src/clip_zeroshot.py](../src/clip_zeroshot.py) | 일반화 — `AutoProcessor`/`AutoModel`로 OpenAI CLIP / MetaCLIP / SigLIP 지원, SigLIP 자동 sigmoid scoring (backward compat 유지) |
| 2026-05-10 | [exp_clip_main.py](../exp_clip_main.py) / [11_e2e_pipeline_v2.py](../11_e2e_pipeline_v2.py) | `--model-id` 인자 추가 — 모델 swap만으로 SigLIP/MetaCLIP 비교 가능 |
| 2026-05-10 | [docs/PIPELINE_HISTORY.md](PIPELINE_HISTORY.md) | 신규 — 본 문서 + v2 결과/설계원칙/multi-model 지원 반영 |
| 2026-05-11 | [exp_clip_direct_sub.py](../exp_clip_direct_sub.py) | 신규 — SigLIP2-L-512 직접 54-way sub 분류 실험 (§ 2.9) |
| 2026-05-11 | [exp_probe_embed.py](../exp_probe_embed.py) | 신규 — SigLIP2-L-512 임베딩 배치 추출 (Step 5) |
| 2026-05-11 | [exp_probe_train.py](../exp_probe_train.py) | 신규 — per-main linear probe 학습 + 평가 (Step 5) |
| 2026-05-11 | [src/clip_zeroshot.py](../src/clip_zeroshot.py) | `encode_image()` 추가 — SigLIP2 ModelOutput 호환 (pooler_output 분기) |
| 2026-05-11 | [11_e2e_pipeline_v2.py](../11_e2e_pipeline_v2.py) | `--probe-dir` 인자 + `_stage_c_probe()` 추가 — Stage C probe 통합 |

## 부록 B. 실험 산출물 위치

- 실험 로그: [experiments/](../experiments/) (`{exp_name}.log`, `{exp_name}_eval.txt`)
- 실험 요약 CSV: [experiments/summary.csv](../experiments/summary.csv)
- E2E 출력 JSONL: [outputs/](../outputs/)
- Crop 결과: [outputs/crops_e2e/](../outputs/crops_e2e/)
- exp1 (sub 분류 비교): [results/exp1/](../results/exp1/)
- DINO 검출 평가: [results/dino_eval/](../results/dino_eval/)
