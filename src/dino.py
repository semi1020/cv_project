"""Grounding DINO wrapper for object detection and crop extraction."""
from pathlib import Path

import torch
from PIL import Image

MODEL_ID = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.50
CROP_PAD_RATIO = 0.15


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
    kept = []
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

    def detect(self, image: Image.Image, text_prompt: str) -> list[dict]:
        """Return list of {box: (x0,y0,x1,y1), score: float} after NMS."""
        inputs = self._proc(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._proc.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        w, h = image.size
        dets = []
        for score, _, box in zip(results["scores"], results["labels"], results["boxes"]):
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0:
                dets.append({"box": (x0, y0, x1, y1), "score": float(score)})
        return _nms(dets, NMS_IOU_THRESHOLD)

    def best_crop(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[Image.Image, bool]:
        """Return (crop, is_fallback). Fallback = full image when no detection."""
        dets = self.detect(image, text_prompt)
        if not dets:
            return image.copy(), True

        best = max(dets, key=lambda d: d["score"])
        x0, y0, x1, y1 = best["box"]
        w, h = image.size
        bw, bh = x1 - x0, y1 - y0
        px, py = int(bw * CROP_PAD_RATIO), int(bh * CROP_PAD_RATIO)
        crop = image.crop((
            max(0, x0 - px), max(0, y0 - py),
            min(w, x1 + px), min(h, y1 + py),
        ))
        return crop, False
