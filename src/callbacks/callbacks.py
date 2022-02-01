import os
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class VisualExtractionCacheCallback(Callback):
    """A simple callback used for caching predictions."""

    def __init__(self, cache_dir: str = "image_cache", cache_suffix: str = ".pt") -> None:
        self.cache_dir = cache_dir
        self.cache_suffix = cache_suffix

        os.makedirs(self.cache_dir, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Store the predictions for a batch."""
        for ex_id, output in outputs.items():
            cache_fname = os.path.join(
                self.cache_dir, os.path.splitext(ex_id)[0] + self.cache_suffix
            )
            features_dict = {
                "bbox_features": output["bbox_features"].to("cpu"),
                "bbox_coords": output["bbox_coords"].to("cpu"),
                "bbox_probas": output["bbox_probas"].to("cpu"),
                "cnn_features": output["cnn_features"].to("cpu"),
            }
            torch.save(features_dict, cache_fname)
