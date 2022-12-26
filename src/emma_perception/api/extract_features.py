from typing import Union

import torch
from PIL import Image
from torch.utils.data import DataLoader

from emma_perception.api.api_dataset import ApiDataset
from emma_perception.api.datamodels import ApiStore
from emma_perception.api.instrumentation import get_tracer
from emma_perception.constants import OBJECT_CLASSMAP
from emma_perception.datamodels import ExtractedFeaturesAPI
from emma_perception.models.vinvl_extractor import VinVLExtractor


tracer = get_tracer(__name__)


@tracer.start_as_current_span("Extract features for one batch")
@torch.inference_mode()
def get_batch_features(
    extractor: VinVLExtractor, batch: dict[str, torch.Tensor], device: torch.device
) -> list[ExtractedFeaturesAPI]:
    """Low-level implementation of the visual feature extraction process."""
    with tracer.start_as_current_span("Inference step"):
        cnn_features: list[torch.Tensor] = []
        hook = extractor.extractor.backbone.register_forward_hook(
            lambda module, inp, output: cnn_features.append(output)
        )

        batch_predictions = extractor.extractor(batch["img"].to(device))
        hook.remove()
        cnn_feats = cnn_features[0][0]

    batch_features: list[ExtractedFeaturesAPI] = []

    with tracer.start_as_current_span("Post-process batch"):
        for batch_idx, predictions in enumerate(batch_predictions):
            # assumes that all the images have the same size
            predictions = predictions.resize(
                (batch["width"][batch_idx], batch["height"][batch_idx])
            )

            bbox_probas = predictions.get_field("scores_all")
            idx_labels = bbox_probas.argmax(dim=1).tolist()
            class_labels = [OBJECT_CLASSMAP["idx_to_label"][str(idx)] for idx in idx_labels]

            features = ExtractedFeaturesAPI(
                bbox_features=predictions.get_field("box_features").tolist(),
                bbox_coords=predictions.bbox.tolist(),
                bbox_probas=bbox_probas.tolist(),
                cnn_features=cnn_feats[batch_idx].mean(dim=(-2, -1)).tolist(),
                class_labels=class_labels,
                width=int(batch["width"][batch_idx].item()),
                height=int(batch["height"][batch_idx].item()),
            )

            batch_features.append(features)

    return batch_features


@tracer.start_as_current_span("Extract all features")
def extract_features_for_batch(
    images: Union[Image.Image, list[Image.Image]], api_store: ApiStore, batch_size: int = 2
) -> list[ExtractedFeaturesAPI]:
    """Extracts visual features for a batch of images."""
    with tracer.start_as_current_span("Build DataLoader"):
        dataset = ApiDataset(images, transform=api_store.transform)
        loader = DataLoader(dataset, batch_size=batch_size)

    all_features: list[ExtractedFeaturesAPI] = []

    for batch in loader:
        all_features.extend(get_batch_features(api_store.extractor, batch, api_store.device))

    return all_features
