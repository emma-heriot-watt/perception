from typing import Any

import torch
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from overrides import overrides
from pytorch_lightning import LightningModule
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.scene_parser import SceneParser


class VinVLTransform:
    """VinVL preprocessing transform."""

    def __init__(self, cfg: Any) -> None:
        self.transforms = build_transforms(cfg, is_train=False)

    def __call__(self, img: Any) -> torch.Tensor:
        """Apply preprocessing transformations to image."""
        img_transf, _ = self.transforms(img, target=None)
        return img_transf


class VinVLExtractor(LightningModule):
    """VinVL object detector wrapped in a lightining module."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()

        if "OUTPUT_FEATURE" not in cfg.TEST and cfg.TEST.OUTPUT_FEATURE:
            raise AttributeError("TEST.OUTPUT_FEATURE must be set to True on configuration file")

        if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
            model = SceneParser(cfg)
        elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
            model = AttrRCNN(cfg)

        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(cfg.MODEL.WEIGHT)

        self.extractor = model

    @overrides(check_signature=False)
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Extracts features, coordinates, and class probabilities for bounding boxes."""
        output = {}
        cnn_features = []

        hook = self.extractor.backbone.register_forward_hook(
            lambda module, inp, output: cnn_features.append(output)
        )
        predictions = self.extractor(batch["img"].to(self.device))
        hook.remove()
        cnn_features = cnn_features[0][0]
        out = zip(batch["ids"], batch["width"], batch["height"], predictions, cnn_features)

        for b_id, width, height, pred, cnn_feats in out:
            pred = pred.resize((width, height))
            output[b_id] = {
                "bbox_features": pred.get_field("box_features"),
                "bbox_coords": pred.bbox,
                "bbox_probas": pred.get_field("scores_all"),
                "cnn_features": cnn_feats.mean(dim=(-2, -1)),
                "width": width,
                "height": height,
            }
        return output
