from typing import Optional

import torch
from pydantic import BaseModel


class ExtractedFeatures(BaseModel, arbitrary_types_allowed=True):
    """Datamodel for returning extracted features."""

    bbox_features: torch.Tensor
    bbox_coords: torch.Tensor
    bbox_probas: torch.Tensor
    cnn_features: torch.Tensor
    entity_labels: Optional[list[str]] = None
    class_labels: list[str]
    width: int
    height: int
