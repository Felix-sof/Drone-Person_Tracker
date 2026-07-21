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
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed_batch(self, crops_bgr: list) -> np.ndarray:
        if not crops_bgr:
            return np.empty((0, 512), dtype=np.float32)
        tensors = torch.stack([
            self.preprocess(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr
        ]).to(self.device)
        features = self.model(tensors).cpu().numpy()
        return features


class _OSNetBackend:
    """Real Re-ID backend via torchreid's pretrained OSNet."""

    def __init__(self, device: str):
        from torchreid.utils import FeatureExtractor
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="",   # empty -> torchreid downloads ImageNet+Re-ID pretrained weights
            device=device,
        )

    def embed_batch(self, crops_bgr: list) -> np.ndarray:
        if not crops_bgr:
            return np.empty((0, 512), dtype=np.float32)
        crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
        features = self.extractor(crops_rgb)  # returns a torch.Tensor
        return features.cpu().numpy()


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
