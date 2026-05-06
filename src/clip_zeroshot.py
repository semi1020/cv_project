"""CLIP zero-shot classifier."""
import torch
from PIL import Image

MODEL_ID = "openai/clip-vit-large-patch14"


class CLIPZeroShot:
    def __init__(self, model_id: str = MODEL_ID):
        from transformers import CLIPModel, CLIPProcessor
        try:
            self._proc = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            self._model = CLIPModel.from_pretrained(model_id, local_files_only=True)
        except Exception:
            self._proc = CLIPProcessor.from_pretrained(model_id)
            self._model = CLIPModel.from_pretrained(model_id)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).eval()

    def classify(
        self, image: Image.Image, candidates: dict[str, str]
    ) -> dict:
        """Classify image against text candidates.

        Args:
            image: PIL image (full or cropped)
            candidates: {label: text_description}

        Returns:
            {"pred": label, "score": float, "all_scores": {label: float}}
        """
        labels = list(candidates.keys())
        texts = list(candidates.values())

        inputs = self._proc(
            text=texts, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu().tolist()

        all_scores = {label: float(score) for label, score in zip(labels, probs)}
        pred = max(all_scores, key=lambda k: all_scores[k])
        return {"pred": pred, "score": all_scores[pred], "all_scores": all_scores}
