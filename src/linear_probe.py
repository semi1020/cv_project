"""CLIP Linear Probe: frozen CLIP encoder + single linear classification head."""
import torch
import torch.nn as nn

MODEL_ID = "openai/clip-vit-large-patch14"


class CLIPLinearProbe(nn.Module):
    def __init__(self, num_classes: int, model_id: str = MODEL_ID):
        super().__init__()
        from transformers import CLIPVisionModel
        self.encoder = CLIPVisionModel.from_pretrained(model_id)
        self.hidden_size = self.encoder.config.hidden_size  # 1024

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(self.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.encoder(pixel_values).pooler_output  # (B, 1024)
        return self.classifier(embeddings)  # (B, num_classes)

    def count_parameters(self) -> dict:
        learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.encoder.parameters())
        return {
            "learnable": learnable,
            "frozen": frozen,
            "ratio": f"{100 * learnable / (frozen + learnable):.4f}%",
        }
