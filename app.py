"""Flask demo server — e2e_v2_siglip2_probe pipeline."""
from __future__ import annotations

import importlib
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from exp_clip_main import build_candidates_canonical
from src.clip_zeroshot import CLIPZeroShot

e2e = importlib.import_module("11_e2e_pipeline_v2")

MODEL_ID  = "google/siglip2-large-patch16-512"
PROBE_DIR = Path("runs/probes")
TOP_K     = 5

# sub_category key → (수수료, CSV 규격명)
FEE_MAP: dict[str, dict] = {
    # TV장식장(거실장)
    "TV장식장(거실장)_가로90㎝이상":              {"fee": 10000, "규격": "가로 90㎝ 이상"},
    "TV장식장(거실장)_단순받침대":                 {"fee":  2000, "규격": "단순받침대"},
    # 가방
    "가방_골프가방":                              {"fee":  3000, "규격": "골프가방"},
    "가방_캐리어":                                {"fee":  2000, "규격": "캐리어"},
    # 거울
    "거울(액자형)_1㎡미만":                       {"fee":  1000, "규격": "1㎡ 미만"},
    "거울(액자형)_1㎡이상":                       {"fee":  2000, "규격": "1㎡ 이상"},
    # 공기청정기
    "공기청정기및가습기_높이1m미만":               {"fee":  3000, "규격": "높이 1m 미만"},
    "공기청정기및가습기_높이1m이상":               {"fee":  5000, "규격": "높이 1m 이상"},
    # 냉장고
    "냉장고_300ℓ미만":                           {"fee":  5000, "규격": "300ℓ 미만"},
    "냉장고_300ℓ이상":                           {"fee":  8000, "규격": "300ℓ 이상"},
    "냉장고_500ℓ이상":                           {"fee": 10000, "규격": "500ℓ 이상"},
    # 상
    "상_4인용미만":                               {"fee":  2000, "규격": "4인용 미만"},
    "상_4인용이상":                               {"fee":  4000, "규격": "4인용 이상"},
    # 소파
    "소파_1인용":                                 {"fee":  5000, "규격": "1인용"},
    "소파_2인용":                                 {"fee":  7000, "규격": "2인용"},
    "소파_3인용이상":                             {"fee":  9000, "규격": "3인용 이상"},
    "소파_카우치":                                {"fee":  6000, "규격": "카우치"},
    "소파_스툴,코너":                             {"fee":  3000, "규격": "스툴+코너"},
    # 소화기
    "소화기_3.5㎏이하(약제기준)":                 {"fee":  2000, "규격": "3.5㎏ 이하"},
    "소화기_3.5㎏초과(약제기준)":                 {"fee":  3000, "규격": "3.5㎏ 초과"},
    # 실내조명등기구
    "실내조명등기구_일반":                         {"fee":  1000, "규격": "일반"},
    "실내조명등기구_장식용":                       {"fee":  3000, "규격": "장식용"},
    # 에어컨및온풍기
    "에어컨및온풍기_0.5㎡미만":                   {"fee":  4000, "규격": "0.5㎡ 미만"},
    "에어컨및온풍기_1.0㎡미만":                   {"fee":  7000, "규격": "1.0㎡ 미만"},
    "에어컨및온풍기_1.0㎡이상":                   {"fee": 10000, "규격": "1.0㎡ 이상"},
    # 의자
    "의자_보조,간이":                             {"fee":  1000, "규격": "보조+간이"},
    "의자_사무용":                                {"fee":  4000, "규격": "사무용"},
    "의자_편의용(안락,흔들,식탁)":                {"fee":  3000, "규격": "편의용(안락+흔들+식탁)"},
    # 자전거
    "자전거_성인용":                              {"fee":  2000, "규격": "성인용"},
    "자전거_아동용":                              {"fee":  1000, "규격": "아동용"},
    # 진열장
    "진열장(장식장,책장,찬장)_가로90㎝미만":      {"fee":  7000, "규격": "가로 90㎝ 미만"},
    "진열장(장식장,책장,찬장)_가로90㎝이상":      {"fee": 10000, "규격": "가로 90㎝ 이상"},
    # 청소기
    "청소기_가정용(모든규격)":                     {"fee":  2000, "규격": "가정용"},
    "청소기_업소용(모든규격)":                     {"fee":  5000, "규격": "업소용"},
    # 컴퓨터
    "컴퓨터_모니터":                              {"fee":  3000, "규격": "모니터"},
    "컴퓨터_본체":                                {"fee":  4000, "규격": "본체"},
    # 텔레비전
    "텔레비전_30인치미만":                         {"fee":  4000, "규격": "30인치 미만"},
    "텔레비전_30인치이상":                         {"fee":  7000, "규격": "30인치 이상"},
    # 형광등기구
    "형광등기구_길이1m미만":                       {"fee":  1000, "규격": "길이 1m 미만"},
    "형광등기구_길이1m이상":                       {"fee":  2000, "규격": "길이 1m 이상"},
    # 단일 sub 클래스
    "선풍기_모든규격":                             {"fee":  2000, "규격": "모든 규격"},
    "세면대(양변기)_모든규격":                     {"fee":  2000, "규격": "모든 규격"},
    "세탁기_8㎏이상":                             {"fee":  7000, "규격": "8㎏ 이상"},
    "시계_벽걸이용":                              {"fee":  1000, "규격": "벽걸이용"},
    "식탁_4인용이상(일반)":                        {"fee":  7000, "규격": "4인용 이상(일반)"},
    "오락기_소형":                                {"fee":  3000, "규격": "소형"},
    "옷걸이_모든규격":                             {"fee":  2000, "규격": "모든 규격"},
    "의료기_일반":                                {"fee":  5000, "규격": "일반"},
    "전기밥솥_모든규격":                           {"fee":  3000, "규격": "모든 규격"},
    "전자레인지_모든규격":                          {"fee":  3000, "규격": "모든 규격"},
    "책상_가로120㎝미만":                          {"fee":  5000, "규격": "가로 120㎝ 미만"},
    "프린트기_업체용대형외모든규격":                {"fee":  2000, "규격": "업체용 대형 외 모든 규격"},
    "피아노_전자피아노,풍금":                       {"fee": 10000, "규격": "전자피아노+풍금"},
    "화장대_가로90㎝미만":                         {"fee":  5000, "규격": "가로 90㎝ 미만"},
}

app = Flask(__name__)

print("[demo] Loading SigLIP2-L-512...", flush=True)
_clf        = CLIPZeroShot(model_id=MODEL_ID)
_candidates = build_candidates_canonical()

_probes: dict = {}
_probe_path = PROBE_DIR / "sub_probes.pt"
if _probe_path.exists():
    _probes = torch.load(_probe_path, weights_only=True, map_location="cpu")
    print(f"[demo] Loaded {len(_probes)} probes", flush=True)
else:
    print(f"[demo] No probes at {_probe_path} — using CLIP zero-shot for Stage C", flush=True)

print("[demo] Ready — http://localhost:5000", flush=True)


@app.route("/")
def index():
    return render_template("index.html")


def _enrich_stage_c(stage_c: dict) -> dict:
    if stage_c.get("all_scores"):
        stage_c["all_scores_sorted"] = sorted(
            stage_c["all_scores"].items(), key=lambda x: x[1], reverse=True
        )
    return stage_c


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "이미지가 없습니다"}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"이미지를 열 수 없습니다: {exc}"}), 400

    # ── Stage A (공유) ────────────────────────────────────────────────────────
    a_raw     = _clf.classify(image, _candidates)
    pred_main = a_raw["pred"]
    topk      = sorted(a_raw["all_scores"].items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    stage_a   = {
        "pred_main": pred_main,
        "score":     round(a_raw["score"], 4),
        "topk":      [{"label": k, "score": round(s, 4)} for k, s in topk],
    }

    # ── Stage C — CLIP zero-shot only (no_b 방식) ────────────────────────────
    c_clip = e2e._stage_c(_clf, image, pred_main)
    c_clip.setdefault("via_probe", False)
    _enrich_stage_c(c_clip)

    # ── Stage C — probe (probe 방식) ─────────────────────────────────────────
    if pred_main in _probes:
        c_probe = e2e._stage_c_probe(_clf, image, _probes[pred_main])
    else:
        c_probe = e2e._stage_c(_clf, image, pred_main)
        c_probe.setdefault("via_probe", False)
    _enrich_stage_c(c_probe)

    def fee(stage_c):
        sub = stage_c.get("pred_sub")
        return FEE_MAP.get(sub) if sub else None

    return jsonify({
        "stage_a":    stage_a,
        "clip":       {"stage_c": c_clip,  "fee": fee(c_clip)},
        "probe":      {"stage_c": c_probe, "fee": fee(c_probe)},
        "image_size": list(image.size),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
