"""Grounding DINO wrapper for object detection and crop extraction."""
from __future__ import annotations

import torch
from PIL import Image

MODEL_ID = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.15
NMS_IOU_THRESHOLD = 0.50
CROP_PAD_RATIO = 0.15
BERT_TOKEN_LIMIT = 256


def _box_iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    ub = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    if len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: list[dict] = []
    for det in ordered:
        if all(_box_iou(det["box"], k["box"]) <= iou_threshold for k in kept):
            kept.append(det)
    return kept


class DINODetector:
    def __init__(self, model_id: str = MODEL_ID):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        try:
            self._proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, local_files_only=True
            )
        except Exception:
            self._proc = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).eval()

    # ------------------------------------------------------------------
    # Prompt budget check (Grounding DINO uses BERT, hard-capped at 256).
    # E2E single-prompt mode (Stage A, ~31 classes with aliases) MUST verify
    # to avoid silent truncation.
    # ------------------------------------------------------------------
    def verify_prompt_budget(self, text_prompt: str, limit: int = BERT_TOKEN_LIMIT) -> int:
        n = len(self._proc.tokenizer.encode(text_prompt))
        if n > limit:
            raise RuntimeError(
                f"Prompt is {n} tokens, exceeds BERT limit {limit}. "
                f"Reduce aliases or chunk the prompt."
            )
        return n

    # ------------------------------------------------------------------
    # Detection (low level). with_labels=True keeps the matched English
    # phrase (label_en) — required for Stage A multi-class single-prompt
    # main classification.
    # ------------------------------------------------------------------
    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        with_labels: bool = False,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> list[dict]:
        """Return list of detections after NMS, sorted by score desc.

        Each dict: {"box": (x0,y0,x1,y1), "score": float[, "label_en": str]}.
        """
        inputs = self._proc(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._proc.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        # Newer transformers exposes "text_labels"; older uses "labels".
        label_field = "text_labels" if "text_labels" in results else "labels"
        w, h = image.size
        dets: list[dict] = []
        for score, label, box in zip(
            results["scores"], results[label_field], results["boxes"]
        ):
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            d: dict = {"box": (x0, y0, x1, y1), "score": float(score)}
            if with_labels:
                d["label_en"] = label
            dets.append(d)
        kept = _nms(dets, NMS_IOU_THRESHOLD)
        kept.sort(key=lambda d: -d["score"])
        return kept

    # ------------------------------------------------------------------
    # Stage B helper: per-main crop extraction.
    # Returns (crop, meta). meta carries detection metadata for downstream
    # CLIP code (dino_meta on SampleRecord).
    # ------------------------------------------------------------------
    def best_crop(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[Image.Image, dict]:
        dets = self.detect(image, text_prompt, with_labels=True)
        w, h = image.size
        if not dets:
            return image.copy(), {
                "detection_success": False,
                "fallback": True,
                "score": None,
                "box": None,
                "label_en": None,
                "image_size": [w, h],
            }

        best = dets[0]
        x0, y0, x1, y1 = best["box"]
        bw, bh = x1 - x0, y1 - y0
        px, py = int(bw * CROP_PAD_RATIO), int(bh * CROP_PAD_RATIO)
        crop_box = (
            max(0, x0 - px), max(0, y0 - py),
            min(w, x1 + px), min(h, y1 + py),
        )
        crop = image.crop(crop_box)
        return crop, {
            "detection_success": True,
            "fallback": False,
            "score": round(best["score"], 4),
            "box": [x0, y0, x1, y1],
            "label_en": best.get("label_en"),
            "image_size": [w, h],
        }
