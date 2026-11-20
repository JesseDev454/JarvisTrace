"""Reusable OpenCLIP encoder extracted from the preserved visual-search baseline."""
from __future__ import annotations
from typing import Optional, Protocol, Sequence
import numpy as np
from PIL import Image
import torch

DEFAULT_CLIP_MODEL_NAME = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b79k"

def make_model_identifier(model_name: str, pretrained: str) -> str:
    """Return a canonical identifier string for an OpenCLIP model and pretrained tag."""
    return f"openclip:{model_name.strip()}:{pretrained.strip()}"


def normalize_embedding_vector(vec: np.ndarray | Sequence[float]) -> np.ndarray:
    """L2-normalize a 1D or 2D embedding array so cosine similarity equals dot product."""
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            raise ValueError("Cannot L2-normalize a zero-norm embedding vector.")
        return (arr / norm).astype(np.float32)
    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("Cannot L2-normalize a batch containing a zero-norm embedding vector.")
        return (arr / norms).astype(np.float32)
    raise ValueError(f"Expected 1D or 2D array for normalization, got shape {arr.shape}")


class VisualTextEncoder(Protocol):
    """Protocol for visual/text encoders (enables both real OpenCLIP and lightweight unit-test encoders)."""

    @property
    def model_id(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    @property
    def device_str(self) -> str: ...

    def encode_images(self, pil_images: Sequence[Image.Image]) -> np.ndarray: ...

    def encode_text(self, query_text: str) -> np.ndarray: ...


class OpenCLIPEmbedder:
    """Singleton-cached OpenCLIP image and text encoder running on CUDA when available."""

    _instance_cache: dict[tuple[str, str, str], "OpenCLIPEmbedder"] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL_NAME,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        device: Optional[str] = None,
    ) -> None:
        import open_clip

        self.model_name = model_name
        self.pretrained = pretrained
        if device is not None:
            resolved_dev = str(device)
        else:
            resolved_dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(resolved_dev)
        self._model_id = make_model_identifier(model_name, pretrained)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self._device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)

        # Determine output embedding dimension with a dummy forward pass or attribute inspection
        with torch.no_grad():
            dummy_tokens = self._tokenizer(["test"]).to(self._device)
            dummy_feat = self._model.encode_text(dummy_tokens)
            self._embedding_dim = int(dummy_feat.shape[-1])

    @classmethod
    def get_instance(
        cls,
        model_name: str = DEFAULT_CLIP_MODEL_NAME,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        device: Optional[str] = None,
    ) -> "OpenCLIPEmbedder":
        """Load the OpenCLIP model once per (model_name, pretrained, device) key."""
        resolved_dev = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = (model_name, pretrained, resolved_dev)
        if cache_key not in cls._instance_cache:
            cls._instance_cache[cache_key] = cls(
                model_name=model_name,
                pretrained=pretrained,
                device=resolved_dev,
            )
        return cls._instance_cache[cache_key]

    @property
    def preprocess_description(self) -> str:
        return repr(self._preprocess)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device_str(self) -> str:
        return str(self._device)

    def encode_images(self, pil_images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode one or more PIL RGB images into L2-normalized float32 embeddings without gradients."""
        if not pil_images:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        all_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(pil_images), max(1, batch_size)):
                chunk = pil_images[start : start + max(1, batch_size)]
                tensors = torch.stack([self._preprocess(img.convert("RGB")) for img in chunk]).to(self._device)
                feats = self._model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                all_batches.append(feats.detach().cpu().to(torch.float32).numpy())

        stacked = np.vstack(all_batches)
        return normalize_embedding_vector(stacked)

    def encode_text(self, query_text: str) -> np.ndarray:
        """Encode a natural-language query string into a 1D L2-normalized float32 embedding without gradients."""
        cleaned = str(query_text or "").strip()
        if not cleaned:
            raise ValueError("Search query text cannot be empty.")
        with torch.no_grad():
            tokens = self._tokenizer([cleaned]).to(self._device)
            text_feat = self._model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            vec = text_feat[0].detach().cpu().to(torch.float32).numpy()
        return normalize_embedding_vector(vec)
