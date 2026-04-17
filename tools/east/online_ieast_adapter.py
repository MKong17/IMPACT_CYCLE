from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


if nn is None:  # pragma: no cover
    class OnlineInteractiveAdapter:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for OnlineInteractiveAdapter.")
else:

    class OnlineInteractiveAdapter(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_ratio: float = 0.5,
            source_buckets: int = 16,
        ) -> None:
            super().__init__()
            self.input_dim = int(max(1, input_dim))
            self.hidden_ratio = float(hidden_ratio if hidden_ratio > 0 else 0.5)
            self.source_buckets = int(max(1, source_buckets))
            hidden_dim = int(max(32, round(self.input_dim * self.hidden_ratio)))

            self.input_norm = nn.LayerNorm(self.input_dim)
            self.source_embed = nn.Embedding(self.source_buckets, self.input_dim)
            self.residual_fc1 = nn.Linear(self.input_dim, hidden_dim)
            self.residual_fc2 = nn.Linear(hidden_dim, self.input_dim)
            self.cp_fc1 = nn.Linear(self.input_dim, hidden_dim)
            self.cp_fc2 = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(0.05)

        @staticmethod
        def normalize_source_name(source_name: Optional[str]) -> str:
            raw = str(source_name or "").strip().lower()
            if not raw:
                return "unknown"
            raw = raw.replace("-", "_").replace(" ", "_")
            aliases = {
                "east_backbone": "east_backbone",
                "east": "east_backbone",
                "videomae": "east_backbone",
                "videomaev2": "east_backbone",
                "video_mae": "east_backbone",
                "dinov2_vitb14": "dinov2_vitb14",
                "dinov2": "dinov2_vitb14",
                "resnet50": "resnet50",
            }
            return aliases.get(raw, raw)

        @classmethod
        def source_bucket_id(cls, source_name: Optional[str], buckets: int = 16) -> int:
            key = cls.normalize_source_name(source_name)
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
            return int(digest[:8], 16) % int(max(1, buckets))

        @staticmethod
        def find_vit_interactive_prefixes(state_dict: Dict[str, Any]) -> List[str]:
            prefixes = []
            for key in (state_dict or {}).keys():
                name = str(key)
                if ".interactive_adapter." in name:
                    prefixes.append(name.split(".interactive_adapter.", 1)[0] + ".interactive_adapter")
            return sorted(set(prefixes))

        def forward(self, features: torch.Tensor, source_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
            if features.dim() == 2:
                x = features.unsqueeze(0)
            elif features.dim() == 3:
                x = features
            else:
                raise ValueError(f"Expected features with 2 or 3 dims, got {tuple(features.shape)}")

            bucket_id = self.source_bucket_id(source_name, buckets=self.source_buckets)
            sid = torch.tensor([bucket_id], device=x.device, dtype=torch.long)
            src_bias = self.source_embed(sid).unsqueeze(1)
            h = self.input_norm(x + 0.05 * src_bias)

            residual = self.residual_fc2(self.dropout(F.gelu(self.residual_fc1(h))))
            adapted = x + 0.15 * residual
            adapted = F.normalize(adapted, dim=-1)

            cp_hidden = F.gelu(self.cp_fc1(adapted))
            cp_logits = self.cp_fc2(self.dropout(cp_hidden)).squeeze(-1)
            return {
                "features": adapted,
                "cp_logits": cp_logits,
            }
