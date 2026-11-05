"""OpenCLIP visual & text embedding extraction, ChromaDB persistent vector index, and description-based retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import chromadb
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from src.models import (
    SightingRecord,
    TrackSummary,
    VideoMetadata,
    format_video_timestamp,
    normalize_source_video,
)
from src.retrieval import determine_synchronization_mode, summarize_tracks
from src.storage import SightingStore

# Suppress Windows huggingface_hub symlink warning cleanly
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DEFAULT_CLIP_MODEL_NAME = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_CHROMA_DIR = "outputs/chroma_db"
DEFAULT_CHROMA_COLLECTION = "sighting_visual_embeddings"
VALID_EMBEDDING_VARIANTS = ("crop", "context")


def make_model_identifier(model_name: str, pretrained: str) -> str:
    """Return a canonical identifier string for an OpenCLIP model and pretrained tag."""
    return f"openclip:{model_name.strip()}:{pretrained.strip()}"


def make_embedding_id(
    sighting_id: int,
    variant: str,
    model_id: str,
) -> str:
    """Return a deterministic ChromaDB document ID for a sighting, variant, and model."""
    if variant not in VALID_EMBEDDING_VARIANTS:
        raise ValueError(f"Invalid embedding variant '{variant}'; expected one of {VALID_EMBEDDING_VARIANTS}")
    safe_model = model_id.replace(":", "_").replace("/", "_")
    return f"s{int(sighting_id)}_{variant}_{safe_model}"


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


def load_pil_image_safely(image_path: str | Path | None) -> Optional[Image.Image]:
    """Load an image file as an RGB PIL.Image, returning None if missing, empty, or corrupted."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return None
    try:
        if p.stat().st_size == 0:
            return None
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            rgb.load()
            if rgb.width <= 0 or rgb.height <= 0:
                return None
            return rgb.copy()
    except (OSError, UnidentifiedImageError, ValueError, RuntimeError):
        return None


def extract_context_region_pil(
    frame_rgb: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    expand_ratio: float = 1.25,
    min_context_pad_px: int = 64,
) -> Optional[Image.Image]:
    """Extract a context-preserving region around `bbox_xyxy` from a full frame image.

    Unlike the tight object crop (`pad=16px`), the context region expands around the object by
    `expand_ratio` (and at least `min_context_pad_px` in each direction) so surrounding surfaces
    (e.g., desk, carrier's hands/torso, adjacent equipment) are preserved for relational queries
    like 'person holding a black laptop' or 'laptop sitting on a wooden desk'.
    """
    w, h = frame_rgb.size
    if w <= 0 or h <= 0:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    pad_x = max(float(min_context_pad_px), bw * expand_ratio)
    pad_y = max(float(min_context_pad_px), bh * expand_ratio)

    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(w, int(round(x2 + pad_x)))
    cy2 = min(h, int(round(y2 + pad_y)))

    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame_rgb.crop((cx1, cy1, cx2, cy2))


class VideoFrameContextReader:
    """Reads clean (unannotated) frames from source videos when available, falling back to evidence frames."""

    def __init__(self) -> None:
        self._caps: dict[str, cv2.VideoCapture] = {}

    def get_frame_rgb(
        self,
        source_video: str,
        frame_index: int,
        evidence_frame_path: Optional[str] = None,
    ) -> tuple[Optional[Image.Image], Optional[str]]:
        """Return (PIL RGB image, source_path_used) for a sighting's full frame."""
        norm_video = normalize_source_video(source_video)
        vpath = Path(norm_video)
        if vpath.exists() and vpath.is_file():
            cap = self._caps.get(norm_video)
            if cap is None:
                cap = cv2.VideoCapture(str(vpath))
                if cap.isOpened():
                    self._caps[norm_video] = cap
                else:
                    cap.release()
                    cap = None
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if ok and frame_bgr is not None and frame_bgr.size > 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame_rgb), (evidence_frame_path or norm_video)

        # Fallback to saved evidence frame image on disk
        fallback_img = load_pil_image_safely(evidence_frame_path)
        if fallback_img is not None:
            return fallback_img, str(evidence_frame_path)
        return None, None

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()


@dataclass
class VisualSightingMatch:
    """A single sighting matched by OpenCLIP visual similarity, resolved against authoritative SQLite metadata."""

    sighting: SightingRecord
    track_status: str  # "STRONGLY_SUPPORTED" or "UNCERTAIN_CANDIDATE"
    heuristic_stability_score: float
    best_similarity: float
    best_variant: str  # "crop" or "context"
    crop_similarity: Optional[float] = None
    context_similarity: Optional[float] = None
    matched_evidence_path: Optional[str] = None
    embedding_model_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sighting_id": self.sighting.sighting_id,
            "run_id": self.sighting.run_id,
            "camera_id": self.sighting.camera_id,
            "source_video": self.sighting.source_video,
            "scoped_track_key": self.sighting.scoped_track_key,
            "local_track_id": self.sighting.local_track_id,
            "frame_index": self.sighting.frame_index,
            "timestamp_seconds": round(float(self.sighting.timestamp_seconds), 4),
            "aligned_wall_clock_utc": self.sighting.aligned_wall_clock_utc,
            "aligned_sync_seconds": self.sighting.aligned_sync_seconds,
            "detected_class_name": self.sighting.class_name,
            "detector_confidence": round(float(self.sighting.confidence), 4),
            "track_support_status": self.track_status,
            "heuristic_stability_score": round(float(self.heuristic_stability_score), 4),
            "clip_similarity": round(float(self.best_similarity), 4),
            "best_embedding_variant": self.best_variant,
            "crop_similarity": round(float(self.crop_similarity), 4) if self.crop_similarity is not None else None,
            "context_similarity": (
                round(float(self.context_similarity), 4) if self.context_similarity is not None else None
            ),
            "bbox_xyxy": [round(float(v), 2) for v in self.sighting.bbox_xyxy],
            "evidence_crop_path": self.sighting.evidence_crop_path,
            "evidence_frame_path": self.sighting.evidence_frame_path,
            "matched_evidence_path": self.matched_evidence_path,
            "embedding_model_id": self.embedding_model_id,
        }


@dataclass
class VisualTrackCandidateGroup:
    """Candidate sightings grouped by isolated local track key without merging across tracks or cameras."""

    scoped_track_key: str
    run_id: Optional[int]
    camera_id: str
    source_video: str
    detected_class_name: str
    local_track_id: Optional[int]
    track_support_status: str  # "STRONGLY_SUPPORTED" or "UNCERTAIN_CANDIDATE"
    track_status_reason: str
    heuristic_stability_score: float
    track_mean_detector_confidence: float
    track_max_detector_confidence: float
    track_frame_span: tuple[int, int]
    track_time_span_seconds: tuple[float, float]
    max_clip_similarity: float
    mean_clip_similarity: float
    best_variant: str
    best_match: VisualSightingMatch
    matched_sightings: list[VisualSightingMatch] = field(default_factory=list)
    identity_note: str = (
        "UNVERIFIED_PHYSICAL_IDENTITY: CLIP visual similarity ranks appearance/context resemblance to the text "
        "description within this isolated camera track; it does NOT prove ground-truth physical object identity."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoped_track_key": self.scoped_track_key,
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "source_video": self.source_video,
            "detected_class_name": self.detected_class_name,
            "local_track_id": self.local_track_id,
            "track_support_status": self.track_support_status,
            "track_status_reason": self.track_status_reason,
            "heuristic_stability_score": round(float(self.heuristic_stability_score), 4),
            "track_mean_detector_confidence": round(float(self.track_mean_detector_confidence), 4),
            "track_max_detector_confidence": round(float(self.track_max_detector_confidence), 4),
            "track_frame_span": [self.track_frame_span[0], self.track_frame_span[1]],
            "track_time_span_seconds": [
                round(float(self.track_time_span_seconds[0]), 4),
                round(float(self.track_time_span_seconds[1]), 4),
            ],
            "max_clip_similarity": round(float(self.max_clip_similarity), 4),
            "mean_clip_similarity": round(float(self.mean_clip_similarity), 4),
            "best_embedding_variant": self.best_variant,
            "matched_sightings_count": len(self.matched_sightings),
            "best_match": self.best_match.to_dict(),
            "matched_sightings": [m.to_dict() for m in self.matched_sightings],
            "identity_note": self.identity_note,
        }


@dataclass
class VisualSearchReport:
    """Structured report for natural-language OpenCLIP + ChromaDB visual search."""

    query_text: str
    embedding_model_id: str
    top_k: int
    camera_filters: list[str]
    class_filter: Optional[str]
    variant_filter: Optional[str]
    min_similarity: Optional[float]
    queried_cameras: list[str]
    chronologically_orderable: bool
    synchronization_mode: str
    total_indexed_embeddings_searched: int
    stale_embeddings_excluded: int
    ranked_sighting_matches: list[VisualSightingMatch] = field(default_factory=list)
    track_candidate_groups: list[VisualTrackCandidateGroup] = field(default_factory=list)
    per_camera_track_groups: dict[str, list[VisualTrackCandidateGroup]] = field(default_factory=dict)
    best_strongly_supported_match: Optional[VisualSightingMatch] = None
    latest_strongly_supported_match: Optional[VisualSightingMatch] = None
    uncertainty_and_calibration_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "embedding_model_id": self.embedding_model_id,
            "top_k": self.top_k,
            "camera_filters": self.camera_filters,
            "class_filter": self.class_filter,
            "variant_filter": self.variant_filter,
            "min_similarity": self.min_similarity,
            "queried_cameras": self.queried_cameras,
            "chronologically_orderable": self.chronologically_orderable,
            "synchronization_mode": self.synchronization_mode,
            "total_indexed_embeddings_searched": self.total_indexed_embeddings_searched,
            "stale_embeddings_excluded": self.stale_embeddings_excluded,
            "best_strongly_supported_match": (
                self.best_strongly_supported_match.to_dict()
                if self.best_strongly_supported_match is not None
                else None
            ),
            "latest_strongly_supported_match": (
                self.latest_strongly_supported_match.to_dict()
                if self.latest_strongly_supported_match is not None
                else None
            ),
            "ranked_sighting_matches": [m.to_dict() for m in self.ranked_sighting_matches],
            "track_candidate_groups": [g.to_dict() for g in self.track_candidate_groups],
            "per_camera_track_groups": {
                cam: [g.to_dict() for g in groups]
                for cam, groups in self.per_camera_track_groups.items()
            },
            "uncertainty_and_calibration_notes": self.uncertainty_and_calibration_notes,
        }


class ChromaVisualIndex:
    """Persistent ChromaDB vector index synchronized with the authoritative SQLite SightingStore."""

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        """Delete and recreate the ChromaDB collection from scratch."""
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self._collection.count())

    def prune_stale_embeddings(self, store: SightingStore) -> dict[str, int]:
        """Remove embeddings from ChromaDB whose run_id or sighting_id no longer exists in SQLite."""
        total_before = self.count()
        if total_before == 0:
            return {"before_count": 0, "removed_count": 0, "after_count": 0}

        valid_runs = {m.run_id for m in store.get_ingestion_runs() if m.run_id is not None}
        valid_sightings = {
            s.sighting_id: s.run_id
            for s in store.query_sightings()
            if s.sighting_id is not None
        }

        all_items = self._collection.get(include=["metadatas"])
        ids: list[str] = list(all_items.get("ids") or [])
        metadatas: list[dict[str, Any]] = list(all_items.get("metadatas") or [])

        stale_ids: list[str] = []
        for doc_id, meta in zip(ids, metadatas):
            if not meta:
                stale_ids.append(doc_id)
                continue
            sid = meta.get("sighting_id")
            rid = meta.get("run_id")
            if sid is None or int(sid) not in valid_sightings:
                stale_ids.append(doc_id)
            elif rid is None or int(rid) not in valid_runs or valid_sightings[int(sid)] != int(rid):
                stale_ids.append(doc_id)

        if stale_ids:
            self._collection.delete(ids=stale_ids)

        after_count = self.count()
        return {
            "before_count": total_before,
            "removed_count": len(stale_ids),
            "after_count": after_count,
        }

    def index_sightings_from_store(
        self,
        store: SightingStore,
        encoder: Optional[VisualTextEncoder] = None,
        camera_ids: Optional[Sequence[str]] = None,
        class_name: Optional[str] = None,
        rebuild: bool = False,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Extract crop and context OpenCLIP embeddings for SQLite sightings with evidence and upsert into ChromaDB."""
        enc = encoder or OpenCLIPEmbedder.get_instance()
        if rebuild:
            self.reset_collection()
            pruned_info = {"before_count": 0, "removed_count": 0, "after_count": 0}
        else:
            pruned_info = self.prune_stale_embeddings(store)

        sightings = store.query_sightings(
            class_name=class_name,
            camera_ids=camera_ids,
        )

        frame_reader = VideoFrameContextReader()
        pending_images: list[Image.Image] = []
        pending_ids: list[str] = []
        pending_metadatas: list[dict[str, Any]] = []
        pending_documents: list[str] = []

        sightings_with_evidence = 0
        sightings_indexed = 0
        crop_embeddings_count = 0
        context_embeddings_count = 0
        missing_or_corrupt_files = 0
        skipped_without_evidence_paths = 0

        try:
            for s in sightings:
                if s.sighting_id is None or s.run_id is None:
                    continue
                if not s.evidence_crop_path and not s.evidence_frame_path:
                    skipped_without_evidence_paths += 1
                    continue

                sightings_with_evidence += 1
                added_for_this_sighting = False

                # 1. Tightly cropped object embedding ('crop')
                if s.evidence_crop_path:
                    crop_pil = load_pil_image_safely(s.evidence_crop_path)
                    if crop_pil is not None:
                        doc_id = make_embedding_id(s.sighting_id, "crop", enc.model_id)
                        pending_images.append(crop_pil)
                        pending_ids.append(doc_id)
                        pending_metadatas.append(
                            self._build_metadata_record(
                                sighting=s,
                                variant="crop",
                                evidence_path=s.evidence_crop_path,
                                model_id=enc.model_id,
                                embedding_dim=enc.embedding_dim,
                            )
                        )
                        pending_documents.append(
                            f"{s.camera_id} {s.class_name} crop track={s.scoped_track_key} t={s.timestamp_seconds:.3f}s"
                        )
                        crop_embeddings_count += 1
                        added_for_this_sighting = True
                    else:
                        missing_or_corrupt_files += 1

                # 2. Context-preserving region embedding ('context')
                if s.evidence_frame_path:
                    frame_pil, frame_src_path = frame_reader.get_frame_rgb(
                        source_video=s.source_video,
                        frame_index=s.frame_index,
                        evidence_frame_path=s.evidence_frame_path,
                    )
                    if frame_pil is not None:
                        ctx_pil = extract_context_region_pil(frame_pil, s.bbox_xyxy)
                        if ctx_pil is not None:
                            doc_id = make_embedding_id(s.sighting_id, "context", enc.model_id)
                            pending_images.append(ctx_pil)
                            pending_ids.append(doc_id)
                            pending_metadatas.append(
                                self._build_metadata_record(
                                    sighting=s,
                                    variant="context",
                                    evidence_path=frame_src_path or s.evidence_frame_path,
                                    model_id=enc.model_id,
                                    embedding_dim=enc.embedding_dim,
                                )
                            )
                            pending_documents.append(
                                f"{s.camera_id} {s.class_name} context track={s.scoped_track_key} t={s.timestamp_seconds:.3f}s"
                            )
                            context_embeddings_count += 1
                            added_for_this_sighting = True
                        else:
                            missing_or_corrupt_files += 1
                    else:
                        missing_or_corrupt_files += 1

                if added_for_this_sighting:
                    sightings_indexed += 1

                if len(pending_images) >= batch_size:
                    self._flush_batch(enc, pending_images, pending_ids, pending_metadatas, pending_documents)
                    pending_images.clear()
                    pending_ids.clear()
                    pending_metadatas.clear()
                    pending_documents.clear()

            if pending_images:
                self._flush_batch(enc, pending_images, pending_ids, pending_metadatas, pending_documents)
        finally:
            frame_reader.close()

        return {
            "persist_dir": self.persist_dir.as_posix(),
            "collection_name": self.collection_name,
            "embedding_model_id": enc.model_id,
            "embedding_dim": enc.embedding_dim,
            "device": enc.device_str,
            "total_sqlite_sightings_scanned": len(sightings),
            "sightings_with_evidence_paths": sightings_with_evidence,
            "sightings_without_evidence_paths": skipped_without_evidence_paths,
            "sightings_indexed": sightings_indexed,
            "crop_embeddings_indexed": crop_embeddings_count,
            "context_embeddings_indexed": context_embeddings_count,
            "total_embeddings_upserted": crop_embeddings_count + context_embeddings_count,
            "missing_or_corrupt_evidence_files": missing_or_corrupt_files,
            "stale_embeddings_pruned": pruned_info["removed_count"],
            "collection_total_embeddings": self.count(),
        }

    def _flush_batch(
        self,
        encoder: VisualTextEncoder,
        images: list[Image.Image],
        ids: list[str],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        embeddings_np = encoder.encode_images(images)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings_np.tolist(),
            metadatas=metadatas,
            documents=documents,
        )

    @staticmethod
    def _build_metadata_record(
        sighting: SightingRecord,
        variant: str,
        evidence_path: str,
        model_id: str,
        embedding_dim: int,
    ) -> dict[str, Any]:
        return {
            "sighting_id": int(sighting.sighting_id or 0),
            "run_id": int(sighting.run_id or 0),
            "camera_id": str(sighting.camera_id),
            "source_video": normalize_source_video(sighting.source_video),
            "frame_index": int(sighting.frame_index),
            "timestamp_seconds": round(float(sighting.timestamp_seconds), 4),
            "scoped_track_key": str(sighting.scoped_track_key),
            "local_track_id": int(sighting.local_track_id) if sighting.local_track_id is not None else -1,
            "class_name": str(sighting.class_name).lower(),
            "class_id": int(sighting.class_id),
            "confidence": round(float(sighting.confidence), 4),
            "evidence_image_path": Path(evidence_path).as_posix(),
            "embedding_variant": str(variant),
            "embedding_model_id": str(model_id),
            "embedding_dim": int(embedding_dim),
        }

    def get_index_stats(self, store: Optional[SightingStore] = None) -> dict[str, Any]:
        """Return detailed breakdown of persisted ChromaDB embeddings and synchronization status with SQLite."""
        total = self.count()
        all_items = self._collection.get(include=["metadatas"]) if total > 0 else {"ids": [], "metadatas": []}
        metadatas: list[dict[str, Any]] = list(all_items.get("metadatas") or [])

        by_camera: dict[str, int] = defaultdict(int)
        by_class: dict[str, int] = defaultdict(int)
        by_variant: dict[str, int] = defaultdict(int)
        by_model: dict[str, int] = defaultdict(int)
        distinct_sighting_ids: set[int] = set()
        distinct_run_ids: set[int] = set()

        for m in metadatas:
            if not m:
                continue
            by_camera[str(m.get("camera_id", "unknown"))] += 1
            by_class[str(m.get("class_name", "unknown"))] += 1
            by_variant[str(m.get("embedding_variant", "unknown"))] += 1
            by_model[str(m.get("embedding_model_id", "unknown"))] += 1
            if m.get("sighting_id") is not None:
                distinct_sighting_ids.add(int(m["sighting_id"]))
            if m.get("run_id") is not None:
                distinct_run_ids.add(int(m["run_id"]))

        stale_count = 0
        sqlite_total_sightings = 0
        sqlite_evidence_sightings = 0
        if store is not None:
            all_sql_sightings = store.query_sightings()
            sqlite_total_sightings = len(all_sql_sightings)
            valid_map = {
                s.sighting_id: s.run_id
                for s in all_sql_sightings
                if s.sighting_id is not None
            }
            sqlite_evidence_sightings = sum(
                1 for s in all_sql_sightings if s.evidence_crop_path or s.evidence_frame_path
            )
            for m in metadatas:
                if not m:
                    stale_count += 1
                    continue
                sid = m.get("sighting_id")
                rid = m.get("run_id")
                if sid is None or int(sid) not in valid_map or valid_map[int(sid)] != int(rid or -1):
                    stale_count += 1

        return {
            "persist_dir": self.persist_dir.as_posix(),
            "collection_name": self.collection_name,
            "total_embeddings": total,
            "distinct_sightings_indexed": len(distinct_sighting_ids),
            "distinct_runs_indexed": sorted(distinct_run_ids),
            "by_camera": dict(sorted(by_camera.items())),
            "by_detected_class": dict(sorted(by_class.items())),
            "by_embedding_variant": dict(sorted(by_variant.items())),
            "by_embedding_model": dict(sorted(by_model.items())),
            "sqlite_total_sightings": sqlite_total_sightings,
            "sqlite_sightings_with_saved_evidence": sqlite_evidence_sightings,
            "stale_embeddings_detected": stale_count,
        }

    def search_by_description(
        self,
        query_text: str,
        store: SightingStore,
        encoder: Optional[VisualTextEncoder] = None,
        top_k: int = 10,
        camera_id: Optional[str] = None,
        camera_ids: Optional[Sequence[str]] = None,
        class_name: Optional[str] = None,
        variant: Optional[str] = None,
        min_similarity: Optional[float] = None,
        confirmed_conf_threshold: float = 0.50,
        confirmed_min_track_frames: int = 3,
        auto_prune_stale: bool = True,
    ) -> VisualSearchReport:
        """Search indexed embeddings with a natural-language description and resolve against SQLite."""
        cleaned_query = str(query_text or "").strip()
        if not cleaned_query:
            raise ValueError("Natural-language query text cannot be empty.")
        if variant is not None and variant not in VALID_EMBEDDING_VARIANTS:
            raise ValueError(f"Invalid variant filter '{variant}'; expected one of {VALID_EMBEDDING_VARIANTS}")

        enc = encoder or OpenCLIPEmbedder.get_instance()
        pruned_removed = 0
        if auto_prune_stale:
            prune_res = self.prune_stale_embeddings(store)
            pruned_removed = prune_res["removed_count"]

        active_cameras: list[str] = []
        if camera_ids:
            active_cameras = [c for c in camera_ids if c]
        elif camera_id:
            active_cameras = [camera_id]

        # Build authoritative SQLite lookup and track support summaries across the queried scope
        sql_sightings = store.query_sightings(
            class_name=class_name,
            camera_ids=active_cameras if active_cameras else None,
        )
        sql_metadata: list[VideoMetadata] = store.get_ingestion_runs(
            camera_ids=active_cameras if active_cameras else None,
        )
        sighting_by_id: dict[int, SightingRecord] = {
            int(s.sighting_id): s for s in sql_sightings if s.sighting_id is not None
        }
        valid_run_ids: set[int] = {int(m.run_id) for m in sql_metadata if m.run_id is not None}

        track_summaries, _ = summarize_tracks(
            sql_sightings,
            confirmed_conf_threshold=confirmed_conf_threshold,
            confirmed_min_track_frames=confirmed_min_track_frames,
            video_metadata=sql_metadata,
        )
        track_summary_by_key: dict[str, TrackSummary] = {
            t.scoped_track_key: t for t in track_summaries
        }

        chronologically_orderable, sync_mode = determine_synchronization_mode(sql_sightings, sql_metadata)
        queried_cameras = sorted(
            {m.camera_id for m in sql_metadata}
            | {s.camera_id for s in sql_sightings}
            | set(active_cameras)
        )

        total_in_collection = self.count()
        if total_in_collection == 0 or not sighting_by_id:
            return self._build_empty_search_report(
                query_text=cleaned_query,
                model_id=enc.model_id,
                top_k=top_k,
                active_cameras=active_cameras,
                class_name=class_name,
                variant=variant,
                min_similarity=min_similarity,
                queried_cameras=queried_cameras,
                chronologically_orderable=chronologically_orderable,
                sync_mode=sync_mode,
                total_searched=0,
                stale_excluded=pruned_removed,
            )

        # Build ChromaDB metadata filter including model compatibility enforcement
        where_clauses: list[dict[str, Any]] = [
            {"embedding_model_id": enc.model_id},
            {"embedding_dim": int(enc.embedding_dim)},
        ]
        if len(active_cameras) == 1:
            where_clauses.append({"camera_id": active_cameras[0]})
        elif len(active_cameras) > 1:
            where_clauses.append({"camera_id": {"$in": active_cameras}})

        if class_name:
            where_clauses.append({"class_name": str(class_name).lower()})
        if variant:
            where_clauses.append({"embedding_variant": str(variant)})

        where_filter: dict[str, Any] = (
            where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}
        )

        # Query ChromaDB for all matching candidate embeddings so both crop & context similarities
        # can be aggregated per sighting before slicing top_k sightings
        matching_subset = self._collection.get(where=where_filter, include=[])
        matching_count = len(matching_subset.get("ids") or [])
        if matching_count == 0:
            return self._build_empty_search_report(
                query_text=cleaned_query,
                model_id=enc.model_id,
                top_k=top_k,
                active_cameras=active_cameras,
                class_name=class_name,
                variant=variant,
                min_similarity=min_similarity,
                queried_cameras=queried_cameras,
                chronologically_orderable=chronologically_orderable,
                sync_mode=sync_mode,
                total_searched=0,
                stale_excluded=pruned_removed,
            )

        query_vec = enc.encode_text(cleaned_query)
        raw_results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=matching_count,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        metadatas_list = (raw_results.get("metadatas") or [[]])[0]
        distances_list = (raw_results.get("distances") or [[]])[0]

        # Aggregate crop and context similarity per SQLite sighting_id
        per_sighting_scores: dict[int, dict[str, Any]] = {}
        stale_during_query = pruned_removed

        for meta, dist in zip(metadatas_list, distances_list):
            if not meta:
                continue
            sid = int(meta.get("sighting_id", -1))
            rid = int(meta.get("run_id", -1))
            sql_record = sighting_by_id.get(sid)
            if sql_record is None or sql_record.run_id != rid or (valid_run_ids and rid not in valid_run_ids):
                stale_during_query += 1
                continue

            # ChromaDB cosine distance is (1.0 - cosine_similarity) for unit-normalized vectors
            cosine_sim = round(float(1.0 - float(dist)), 4)
            if min_similarity is not None and cosine_sim < float(min_similarity):
                continue

            emb_variant = str(meta.get("embedding_variant", "crop"))
            ev_path = str(meta.get("evidence_image_path", ""))
            emb_model = str(meta.get("embedding_model_id", enc.model_id))

            entry = per_sighting_scores.get(sid)
            if entry is None:
                entry = {
                    "sighting": sql_record,
                    "crop_similarity": None,
                    "context_similarity": None,
                    "best_similarity": cosine_sim,
                    "best_variant": emb_variant,
                    "matched_evidence_path": ev_path,
                    "embedding_model_id": emb_model,
                }
                per_sighting_scores[sid] = entry

            if emb_variant == "crop":
                entry["crop_similarity"] = cosine_sim
            elif emb_variant == "context":
                entry["context_similarity"] = cosine_sim

            if cosine_sim > entry["best_similarity"]:
                entry["best_similarity"] = cosine_sim
                entry["best_variant"] = emb_variant
                entry["matched_evidence_path"] = ev_path

        all_sighting_matches: list[VisualSightingMatch] = []
        for sid, info in per_sighting_scores.items():
            s_rec: SightingRecord = info["sighting"]
            tsum = track_summary_by_key.get(s_rec.scoped_track_key)
            if tsum is not None:
                # A sighting is STRONGLY_SUPPORTED only if its track is STRONGLY_SUPPORTED
                # AND its frame-level detector confidence meets confirmed_conf_threshold
                is_supp = (
                    tsum.status == "STRONGLY_SUPPORTED"
                    and s_rec.confidence >= confirmed_conf_threshold
                )
                t_status = "STRONGLY_SUPPORTED" if is_supp else "UNCERTAIN_CANDIDATE"
                stab_score = tsum.heuristic_stability_score
            else:
                t_status = "UNCERTAIN_CANDIDATE"
                stab_score = 0.0

            all_sighting_matches.append(
                VisualSightingMatch(
                    sighting=s_rec,
                    track_status=t_status,
                    heuristic_stability_score=stab_score,
                    best_similarity=info["best_similarity"],
                    best_variant=info["best_variant"],
                    crop_similarity=info["crop_similarity"],
                    context_similarity=info["context_similarity"],
                    matched_evidence_path=info["matched_evidence_path"],
                    embedding_model_id=info["embedding_model_id"],
                )
            )

        # Sort sightings by highest CLIP similarity first, breaking ties by detector confidence and timestamp
        all_sighting_matches.sort(
            key=lambda m: (
                -m.best_similarity,
                -m.sighting.confidence,
                m.sighting.timestamp_seconds,
                m.sighting.frame_index,
            )
        )
        ranked_top_k = all_sighting_matches[: max(1, int(top_k))]

        # Group matches by isolated track key (using all matches that passed min_similarity, or top matches)
        by_track_key: dict[str, list[VisualSightingMatch]] = defaultdict(list)
        for m in all_sighting_matches:
            by_track_key[m.sighting.scoped_track_key].append(m)

        track_groups: list[VisualTrackCandidateGroup] = []
        for tkey, t_matches in by_track_key.items():
            t_matches_sorted = sorted(
                t_matches,
                key=lambda m: (-m.best_similarity, -m.sighting.confidence, m.sighting.frame_index),
            )
            best_m = t_matches_sorted[0]
            sims = [m.best_similarity for m in t_matches_sorted]
            tsum = track_summary_by_key.get(tkey)
            s0 = best_m.sighting

            if tsum is not None:
                t_status = tsum.status
                t_reason = tsum.status_reason
                t_stab = tsum.heuristic_stability_score
                t_mean_conf = tsum.mean_confidence
                t_max_conf = tsum.max_confidence
                f_span = (tsum.first_frame, tsum.last_frame)
                time_span = (tsum.first_timestamp_seconds, tsum.last_timestamp_seconds)
            else:
                t_status = "UNCERTAIN_CANDIDATE"
                t_reason = "No track summary available."
                t_stab = 0.0
                t_mean_conf = s0.confidence
                t_max_conf = s0.confidence
                f_span = (s0.frame_index, s0.frame_index)
                time_span = (s0.timestamp_seconds, s0.timestamp_seconds)

            track_groups.append(
                VisualTrackCandidateGroup(
                    scoped_track_key=tkey,
                    run_id=s0.run_id,
                    camera_id=s0.camera_id,
                    source_video=s0.source_video,
                    detected_class_name=s0.class_name,
                    local_track_id=s0.local_track_id,
                    track_support_status=t_status,
                    track_status_reason=t_reason,
                    heuristic_stability_score=t_stab,
                    track_mean_detector_confidence=t_mean_conf,
                    track_max_detector_confidence=t_max_conf,
                    track_frame_span=f_span,
                    track_time_span_seconds=time_span,
                    max_clip_similarity=max(sims),
                    mean_clip_similarity=round(sum(sims) / len(sims), 4),
                    best_variant=best_m.best_variant,
                    best_match=best_m,
                    matched_sightings=t_matches_sorted[:5],
                )
            )

        track_groups.sort(
            key=lambda g: (-g.max_clip_similarity, -g.heuristic_stability_score, g.track_time_span_seconds[0])
        )

        per_camera_groups: dict[str, list[VisualTrackCandidateGroup]] = defaultdict(list)
        for g in track_groups:
            per_camera_groups[g.camera_id].append(g)

        supported_matches = [m for m in all_sighting_matches if m.track_status == "STRONGLY_SUPPORTED"]
        best_supported = supported_matches[0] if supported_matches else None

        latest_supported: Optional[VisualSightingMatch] = None
        if chronologically_orderable and supported_matches:
            # Among strongly supported sightings within the top visual candidate tracks, find the latest in time
            top_supported_tracks = {
                g.scoped_track_key
                for g in track_groups
                if g.track_support_status == "STRONGLY_SUPPORTED"
            }
            relevant_supp = [
                m for m in supported_matches if m.sighting.scoped_track_key in top_supported_tracks
            ]
            if relevant_supp:
                latest_supported = max(
                    relevant_supp,
                    key=lambda m: (
                        m.sighting.aligned_epoch_seconds
                        if sync_mode == "MULTI_CAMERA_WALL_CLOCK_ALIGNED" and m.sighting.aligned_epoch_seconds is not None
                        else (
                            m.sighting.aligned_sync_seconds
                            if sync_mode == "MULTI_CAMERA_OFFSET_ALIGNED" and m.sighting.aligned_sync_seconds is not None
                            else m.sighting.timestamp_seconds
                        ),
                        m.sighting.frame_index,
                    ),
                )

        notes: list[str] = [
            "CLIP SIMILARITY CALIBRATION: Cosine similarity between OpenCLIP image and text embeddings is a "
            "relative visual/semantic ranking score, NOT a calibrated probability of correctness or proof of "
            "physical object identity.",
            "CROP VS. CONTEXT EMBEDDINGS: 'crop' embeddings represent the tightly padded object bounding box, "
            "whereas 'context' embeddings include the surrounding scene region (desk surface, carrier hands/body) "
            "to support relational descriptions.",
            "SEPARATION OF EVIDENCE PROPERTIES: Detector confidence (YOLO), within-camera track stability "
            "(STRONGLY_SUPPORTED vs UNCERTAIN_CANDIDATE), and OpenCLIP visual-text similarity are reported "
            "independently and are never conflated.",
            "NO AUTOMATIC CROSS-TRACK OR CROSS-CAMERA MERGING: Candidate tracks remain strictly isolated by "
            "`run_id:camera_id:source_video:local_track_id`.",
        ]
        if not chronologically_orderable:
            notes.append(
                "UNSYNCHRONIZED CAMERAS: Multiple camera recordings lack shared synchronization metadata; "
                "matches are grouped per camera without cross-camera chronological ordering."
            )
        if not ranked_top_k:
            notes.append(
                f"NO VISUAL MATCHES: 0 indexed sightings met the search criteria for '{cleaned_query}'."
            )

        return VisualSearchReport(
            query_text=cleaned_query,
            embedding_model_id=enc.model_id,
            top_k=top_k,
            camera_filters=active_cameras,
            class_filter=class_name,
            variant_filter=variant,
            min_similarity=min_similarity,
            queried_cameras=queried_cameras,
            chronologically_orderable=chronologically_orderable,
            synchronization_mode=sync_mode,
            total_indexed_embeddings_searched=matching_count,
            stale_embeddings_excluded=stale_during_query,
            ranked_sighting_matches=ranked_top_k,
            track_candidate_groups=track_groups,
            per_camera_track_groups=dict(sorted(per_camera_groups.items())),
            best_strongly_supported_match=best_supported,
            latest_strongly_supported_match=latest_supported,
            uncertainty_and_calibration_notes=notes,
        )

    @staticmethod
    def _build_empty_search_report(
        query_text: str,
        model_id: str,
        top_k: int,
        active_cameras: list[str],
        class_name: Optional[str],
        variant: Optional[str],
        min_similarity: Optional[float],
        queried_cameras: list[str],
        chronologically_orderable: bool,
        sync_mode: str,
        total_searched: int,
        stale_excluded: int,
    ) -> VisualSearchReport:
        return VisualSearchReport(
            query_text=query_text,
            embedding_model_id=model_id,
            top_k=top_k,
            camera_filters=active_cameras,
            class_filter=class_name,
            variant_filter=variant,
            min_similarity=min_similarity,
            queried_cameras=queried_cameras,
            chronologically_orderable=chronologically_orderable,
            synchronization_mode=sync_mode,
            total_indexed_embeddings_searched=total_searched,
            stale_embeddings_excluded=stale_excluded,
            uncertainty_and_calibration_notes=[
                f"NO VISUAL MATCHES: 0 indexed embeddings matched the query '{query_text}' and filter scope.",
                "CLIP SIMILARITY CALIBRATION: Cosine similarity is a relative ranking score, NOT a calibrated "
                "probability or proof of physical identity.",
            ],
        )


def format_index_stats_text(stats: dict[str, Any]) -> str:
    """Format ChromaDB visual embedding index statistics for CLI output."""
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("CHROMADB + OPENCLIP VISUAL EMBEDDING INDEX STATISTICS")
    lines.append("=" * 88)
    lines.append(f"ChromaDB Persist Directory       : {stats['persist_dir']}")
    lines.append(f"Collection Name                  : {stats['collection_name']}")
    lines.append(f"Total Persisted Embeddings       : {stats['total_embeddings']}")
    lines.append(f"Distinct Sightings Indexed       : {stats['distinct_sightings_indexed']}")
    lines.append(f"Distinct Ingestion Runs Indexed  : {stats['distinct_runs_indexed']}")
    lines.append(f"SQLite Authoritative Sightings   : {stats['sqlite_total_sightings']} total ({stats['sqlite_sightings_with_saved_evidence']} with saved evidence images)")
    lines.append(f"Stale Embeddings Detected        : {stats['stale_embeddings_detected']}")
    lines.append("-" * 88)
    lines.append(f"Embeddings by Camera             : {stats['by_camera']}")
    lines.append(f"Embeddings by Detected Class     : {stats['by_detected_class']}")
    lines.append(f"Embeddings by Variant            : {stats['by_embedding_variant']}")
    lines.append(f"Embeddings by OpenCLIP Model     : {stats['by_embedding_model']}")
    lines.append("=" * 88)
    return "\n".join(lines)


def format_visual_search_report_text(report: VisualSearchReport) -> str:
    """Format a human-readable OpenCLIP + ChromaDB natural-language retrieval report."""
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(f"OPENCLIP + CHROMADB DESCRIPTION-BASED VISUAL RETRIEVAL REPORT")
    lines.append("=" * 88)
    lines.append(f"Natural-Language Query   : \"{report.query_text}\"")
    lines.append(f"OpenCLIP Model ID        : {report.embedding_model_id}")
    lines.append(
        f"Filter Scope             : cameras={report.camera_filters or 'ALL'} | "
        f"class={report.class_filter or 'ALL'} | variant={report.variant_filter or 'crop+context'} | "
        f"min_sim={report.min_similarity if report.min_similarity is not None else 'None'}"
    )
    lines.append(
        f"Synchronization Mode     : {report.synchronization_mode} "
        f"(Cross-camera chronological ordering valid: {report.chronologically_orderable})"
    )
    lines.append(
        f"Embeddings Searched      : {report.total_indexed_embeddings_searched} "
        f"(Stale excluded: {report.stale_embeddings_excluded})"
    )

    lines.append("-" * 88)
    lines.append(
        "1. CANDIDATE TRACK GROUPS RANKED BY VISUAL SIMILARITY (ISOLATED TRACK IDENTITIES):"
    )
    if not report.track_candidate_groups:
        lines.append("  [NONE] No candidate tracks matched the query.")
    else:
        for idx, g in enumerate(report.track_candidate_groups, start=1):
            bm = g.best_match
            lines.append(
                f"  [{idx}] {g.scoped_track_key} | Camera='{g.camera_id}' | "
                f"Detected Class='{g.detected_class_name}' | Status={g.track_support_status}"
            )
            lines.append(
                f"      Max CLIP Similarity : {g.max_clip_similarity:.4f} (via '{g.best_variant}' embedding; "
                f"track mean sim={g.mean_clip_similarity:.4f} across { len(g.matched_sightings) } shown match(es))"
            )
            lines.append(
                f"      Detector & Stability: mean_det_conf={g.track_mean_detector_confidence:.2f}, "
                f"max_det_conf={g.track_max_detector_confidence:.2f} | "
                f"heuristic_stability={g.heuristic_stability_score:.2f}"
            )
            lines.append(
                f"      Track Span          : Frames {g.track_frame_span[0]}..{g.track_frame_span[1]} "
                f"({g.track_time_span_seconds[0]:.3f}s -> {g.track_time_span_seconds[1]:.3f}s)"
            )
            lines.append(
                f"      Best Matched Frame  : Sighting #{bm.sighting.sighting_id} @ "
                f"{format_video_timestamp(bm.sighting.timestamp_seconds)} [Frame {bm.sighting.frame_index}] | "
                f"det_conf={bm.sighting.confidence:.4f} | "
                f"crop_sim={f'{bm.crop_similarity:.4f}' if bm.crop_similarity is not None else 'N/A'}, "
                f"context_sim={f'{bm.context_similarity:.4f}' if bm.context_similarity is not None else 'N/A'}"
            )
            lines.append(
                f"      Supporting Evidence : crop={bm.sighting.evidence_crop_path or 'N/A'} | "
                f"frame={bm.sighting.evidence_frame_path or 'N/A'}"
            )

    lines.append("-" * 88)
    lines.append(
        f"2. TOP-{report.top_k} RANKED INDIVIDUAL SIGHTING MATCHES (UNCALIBRATED COSINE SIMILARITY):"
    )
    if not report.ranked_sighting_matches:
        lines.append("  [NONE] No individual sightings matched the query.")
    else:
        for rank, m in enumerate(report.ranked_sighting_matches, start=1):
            s = m.sighting
            c_sim = f"{m.crop_similarity:.4f}" if m.crop_similarity is not None else "  N/A "
            ctx_sim = f"{m.context_similarity:.4f}" if m.context_similarity is not None else "  N/A "
            lines.append(
                f"  #{rank:2d} | sim={m.best_similarity:.4f} ({m.best_variant:7s} | crop={c_sim}, ctx={ctx_sim}) | "
                f"Cam='{s.camera_id}' | {format_video_timestamp(s.timestamp_seconds)} [Frame {s.frame_index:3d}] | "
                f"class='{s.class_name}' det_conf={s.confidence:.4f} [{m.track_status}] | "
                f"{s.scoped_track_key} | crop={s.evidence_crop_path or 'N/A'}"
            )

    lines.append("-" * 88)
    lines.append("3. EVIDENCE & UNCERTAINTY CALIBRATION NOTES:")
    for note in report.uncertainty_and_calibration_notes:
        lines.append(f"  ! {note}")
    lines.append("=" * 88)
    return "\n".join(lines)

