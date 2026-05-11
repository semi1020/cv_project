"""Open-vocabulary zero-shot classifier — CLIP / MetaCLIP / SigLIP family.

Supports any HuggingFace model exposing `logits_per_image` via AutoModel:
  - OpenAI CLIP   : openai/clip-vit-{base,large}-patch{14,16,32}
  - MetaCLIP      : facebook/metaclip-{b16,l14,h14}-*
  - SigLIP        : google/siglip-{base,large,so400m}-patch*
  - SigLIP2       : google/siglip2-*
  - EVA-CLIP      : QuanSun/EVA-CLIP-* (CLIP-compatible)
  - OpenCLIP      : laion/CLIP-* variants on the hub

Default model = OpenAI CLIP ViT-L/14 (역사적 baseline 79.5% main, see PIPELINE_HISTORY § 1).

Scoring:
  - CLIP / MetaCLIP : cosine sim → softmax across candidates (standard ZSC)
  - SigLIP / SigLIP2: cosine sim → sigmoid per candidate (training-aligned)
  Argmax is invariant; only `score` interpretation differs.
"""
import torch
from PIL import Image

DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
# Backward-compat alias used by other modules
MODEL_ID = DEFAULT_MODEL_ID


def _is_siglip_family(model_id: str) -> bool:
    return "siglip" in model_id.lower()


class CLIPZeroShot:
    """Generic open-vocab zero-shot classifier (CLIP / MetaCLIP / SigLIP)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        from transformers import AutoModel, AutoProcessor
        # use_safetensors=True: .bin (pickle) fallback 차단 — torch<2.6 CVE-2025-32434 우회.
        # SigLIP / MetaCLIP / OpenAI CLIP 전부 safetensors 제공.
        model_kwargs = {"use_safetensors": True}
        try:
            self._proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            self._model = AutoModel.from_pretrained(model_id, local_files_only=True, **model_kwargs)
        except Exception:
            self._proc = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModel.from_pretrained(model_id, **model_kwargs)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).eval()
        self.model_id = model_id
        self._is_siglip = _is_siglip_family(model_id)

    def classify(self, image: Image.Image, candidates: dict[str, str]) -> dict:
        """Classify image against text candidates.

        Args:
            image: PIL image (full or cropped).
            candidates: {label: text_description}.

        Returns:
            {"pred": label, "score": float, "all_scores": {label: float}}
        """
        labels = list(candidates.keys())
        texts = list(candidates.values())

        inputs = self._proc(
            text=texts, images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image.squeeze(0)  # (n_texts,)
            if self._is_siglip:
                probs = logits.sigmoid().cpu().tolist()
            else:
                probs = logits.softmax(dim=-1).cpu().tolist()

        all_scores = {label: float(s) for label, s in zip(labels, probs)}
        pred = max(all_scores, key=lambda k: all_scores[k])
        return {"pred": pred, "score": all_scores[pred], "all_scores": all_scores}
