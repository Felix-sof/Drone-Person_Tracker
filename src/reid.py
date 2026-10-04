"""
Re-Identification: turn a person crop into a fixed-length embedding vector,
and compare embeddings via cosine similarity.

Two backends are supported (see config.REID_BACKEND):

  "osnet"    -> a real person Re-ID model (via the `torchreid` package),
                trained specifically to distinguish one person from another
                across viewpoints/lighting. This is what you want for real
                accuracy.
  "resnet18" -> a general ImageNet classifier repurposed as a feature
                extractor. No extra install, works everywhere, but was
                never trained to tell people apart -- only to recognize
                "this is a person" as a category. Kept as a zero-dependency
                fallback so the pipeline still runs if torchreid isn't
                installed (e.g. torchreid failed to build on this machine).
"""

import cv2
import numpy as np
import torch

from config import REID_BACKEND, REID_MATCH_THRESHOLD


_REID_INPUT_HW = (256, 128)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess_batch(crops_bgr: list, device) -> torch.Tensor:
    """
    Person crops -> normalized (N, 3, 256, 128) float tensor on `device`.

    Equivalent to torchvision's ToPILImage -> Resize -> ToTensor ->
    Normalize chain, but much cheaper: each crop is resized with cv2 (no
    PIL round-trip), the whole batch crosses to the GPU ONCE as uint8 (4x
    less data than float32), and the float conversion + normalization run
    on the GPU. With a crowd (dozens of unclaimed people re-embedded every
    frame), the per-crop PIL path was the single biggest cost in the whole
    pipeline -- more than detection itself.
    """
    h, w = _REID_INPUT_HW
    batch = np.stack([
        cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in crops_bgr
    ])                                                          # (N, H, W, 3) BGR uint8
    x = torch.from_numpy(batch).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).flip(1).float().div_(255.0)      # -> (N, 3, H, W) RGB
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class _ResNet18Backend:
    """Fallback backend: general-purpose, not person-specific."""

    def __init__(self, device: str):
        import torch.nn as nn
        import torchvision.transforms as T
        from torchvision.models import resnet18, ResNet18_Weights

        self.device = device
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.model = backbone.to(device).eval()

    @torch.no_grad()
    def embed_batch(self, crops_bgr: list) -> np.ndarray:
        if not crops_bgr:
            return np.empty((0, 512), dtype=np.float32)
        return self.model(_preprocess_batch(crops_bgr, self.device)).float().cpu().numpy()


class _OSNetBackend:
    """Real Re-ID backend via torchreid's pretrained OSNet."""

    def __init__(self, device: str):
        # torchreid moved FeatureExtractor between releases (the PyPI
        # 0.2.5 wheel ships it under torchreid.reid.utils); without the
        # fallback import, OSNet silently never loads and every run uses
        # the much weaker resnet18 backend.
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError:
            from torchreid.reid.utils import FeatureExtractor
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            # Empty -> torchreid downloads ImageNet-pretrained OSNet weights
            # only (NOT a person Re-ID checkpoint such as Market-1501/MSMT17).
            # Point this at a Re-ID-trained .pth from the torchreid model
            # zoo for substantially better identity matching.
            model_path="",
            device=device,
        )

    @torch.no_grad()
    def embed_batch(self, crops_bgr: list) -> np.ndarray:
        if not crops_bgr:
            return np.empty((0, 512), dtype=np.float32)
        # Bypass FeatureExtractor.__call__ (per-crop PIL preprocessing on
        # the CPU) and feed the model directly -- same transform, see
        # _preprocess_batch.
        x = _preprocess_batch(crops_bgr, self.extractor.device)
        return self.extractor.model(x).float().cpu().numpy()


class ReIDEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend_name = REID_BACKEND
        self._backend = self._build_backend(self.backend_name)

    def _build_backend(self, name: str):
        if name == "osnet":
            try:
                return _OSNetBackend(self.device)
            except Exception as exc:
                print(
                    f"[reid] Could not load OSNet backend ({exc}). "
                    "Falling back to resnet18. Run `pip install torchreid` "
                    "and check your internet connection for weight download."
                )
                self.backend_name = "resnet18"
                return _ResNet18Backend(self.device)
        return _ResNet18Backend(self.device)

    def embed(self, person_crop_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([person_crop_bgr])[0]

    def embed_batch(self, person_crops_bgr: list) -> np.ndarray:
        """Returns L2-normalized embeddings, shape (N, D)."""
        features = self._backend.embed_batch(person_crops_bgr)
        if len(features) == 0:
            return features
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both vectors are assumed L2-normalized already."""
    return float(np.dot(a, b))


def best_match(reference_gallery: np.ndarray, candidate_embs: np.ndarray):
    """
    Args:
        reference_gallery: (G, D) normalized vectors -- one or more reference
            shots of the target (e.g. front view, back view, side view).
            A single-shot reference is just G=1.
        candidate_embs: (N, D) normalized vectors for this frame's detections.

    For each candidate, the match score is its BEST similarity against any
    single shot in the gallery -- i.e. "does this candidate look like the
    target from AT LEAST ONE of the angles we've captured", not an average
    across angles (averaging would wash out a strong match to one angle
    with weak similarity to the others).

    Returns:
        (best_index, best_score) or (None, best_score) if below threshold.
    """
    if len(candidate_embs) == 0 or len(reference_gallery) == 0:
        return None, 0.0

    sims = candidate_embs @ reference_gallery.T  # (N, G)
    best_per_candidate = sims.max(axis=1)        # (N,) best angle match each
    best_idx = int(np.argmax(best_per_candidate))
    best_score = float(best_per_candidate[best_idx])

    if best_score >= REID_MATCH_THRESHOLD:
        return best_idx, best_score
    return None, best_score
