from typing import Optional

import torch
from pydantic import BaseModel


class ExtractedFeatures(BaseModel, arbitrary_types_allowed=True):
    """Datamodel for returning extracted features."""

    bbox_features: torch.Tensor
    bbox_coords: torch.Tensor
    bbox_probas: torch.Tensor
    cnn_features: torch.Tensor
    class_labels: list[str]


class ExtractedFeaturesAPI(BaseModel):
    """Datamodel for returning extracted features."""

    bbox_features: list[list[float]]
    bbox_coords: list[list[float]]
    bbox_probas: list[list[float]]
    cnn_features: list[float]
    entity_labels: Optional[list[str]] = None
    class_labels: list[str]
    width: int
    height: int
