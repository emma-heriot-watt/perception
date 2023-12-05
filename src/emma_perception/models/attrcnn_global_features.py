from typing import Optional

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import ImageList
from scene_graph_benchmark.AttrRCNN import AttrRCNN


class AttrRCNNGlobalFeatures(AttrRCNN):  # type: ignore[misc]
    """AttrRCNN model extended to additionally return scene features."""

    def forward(self, images: ImageList, targets: Optional[tuple[BoxList]] = None) -> None:
        """Extract AttrRCNN and scene features."""
        cnn_features = []
        hook = self.backbone.register_forward_hook(
            lambda module, inp, output: cnn_features.append(output)
        )
        predictions = super().forward(images, targets)
        hook.remove()
        cnn_features = cnn_features[0][0]
        for pred, feats in zip(predictions, cnn_features):
            pred.add_field("cnn_features", feats.mean(dim=(-2, -1)))

        return predictions
