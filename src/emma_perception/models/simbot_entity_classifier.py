import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from numpy import typing
from torch.nn import BatchNorm1d, CosineSimilarity, Dropout, Linear, Module, Sequential


class SimBotEntityClassifier:
    """SimBotEntityClassifer class."""

    default_tensor_device: torch.device = torch.device("cpu")

    def __init__(
        self,
        class_centroids_path: Path,
        device: torch.device = default_tensor_device,
    ) -> None:
        self.device = device
        (centroids_vectors, centroids_names) = self._load_centroids(class_centroids_path)
        self.centroids_vectors = centroids_vectors
        self.centroids_names = centroids_names
        self._cosine_distance = CosineSimilarity()

    def __call__(self, bbox_features: torch.Tensor, class_labels: list[str]) -> list[str]:
        """Find the object entity for all detected objects."""
        entities = []
        for idx, class_label in enumerate(class_labels):
            centroid_vectors = self.centroids_vectors.get(class_label, None)
            if centroid_vectors is not None:
                entity_index = self._cosine_distance(bbox_features[idx], centroid_vectors).argmax()
                # TODO: remove this with the new entity classifier
                entity = self.centroids_names[class_label][entity_index]
                if entity == "Round Table":
                    entity = "Table"
                entities.append(entity)
            else:
                entities.append(class_label)
        return entities

    def move_to_device(self, device: torch.device) -> None:
        """Moves all the centroids to the specified device."""
        for key, tensor in self.centroids_vectors.items():
            self.centroids_vectors[key] = tensor.to(device)

    def _load_centroids(
        self, class_centroids_path: Path
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
        centroids = torch.load(class_centroids_path, map_location=self.device)
        object_class_centroids_vectors = {}
        object_class_centroids_names = {}
        for object_class, centroid_dict in centroids.items():
            vectors = torch.stack(list(centroid_dict.values()))
            names_list = list(centroid_dict.keys())
            object_class_centroids_vectors[object_class] = vectors
            object_class_centroids_names[object_class] = names_list
        return object_class_centroids_vectors, object_class_centroids_names


class SimBotKNNEntityClassifier:
    """SimBotKNNEntityClassifier class."""

    default_tensor_device: torch.device = torch.device("cpu")

    def __init__(
        self,
        class_centroids_path: Path,
        device: torch.device = default_tensor_device,
        neighbors: int = 5,
    ) -> None:
        self.device = device
        (centroids_vectors, centroids_names) = self._load_centroids(class_centroids_path)
        self.centroids_vectors = centroids_vectors
        self.centroids_names = centroids_names
        self._cosine_distance = CosineSimilarity()
        self._neighbors = neighbors

    def __call__(self, bbox_features: torch.Tensor, class_labels: list[str]) -> list[str]:
        """Find the object entity for all detected objects."""
        entities = []
        for idx, class_label in enumerate(class_labels):
            centroid_vectors = self.centroids_vectors.get(class_label, None)
            if centroid_vectors is not None:
                distances = self._cosine_distance(bbox_features[idx], centroid_vectors)
                knn = distances.topk(self._neighbors, largest=True)
                labels = self.centroids_names[class_label][knn.indices.tolist()]
                entity: str = Counter(labels).most_common(1)[0][0]

                # TODO: remove this with the new entity classifier
                if entity == "Round Table":
                    entity = "Table"
                entities.append(entity)
            else:
                entities.append(class_label)
        return entities

    def move_to_device(self, device: torch.device) -> None:
        """Moves all the centroids to the specified device."""
        for key, tensor in self.centroids_vectors.items():
            self.centroids_vectors[key] = tensor.to(device)

    def _load_centroids(
        self, class_centroids_path: Path
    ) -> tuple[dict[str, torch.Tensor], dict[str, typing.NDArray[np.str_]]]:
        centroids = torch.load(class_centroids_path, map_location=self.device)
        centroid_vectors = {}
        centroid_names = {}
        for object_class, object_dict in centroids.items():
            centroid_vectors[object_class] = object_dict["features"]
            centroid_names[object_class] = np.array(object_dict["labels"])
        return centroid_vectors, centroid_names


class SimBotMLPEntityClassifier:
    """SimBotMLPEntityClassifier class."""

    default_tensor_device: torch.device = torch.device("cpu")

    def __init__(
        self,
        model_path: Path,
        classmap_path: Path,
        device: torch.device = default_tensor_device,
    ) -> None:
        self.device = device
        self.classifier = EntityPolicy.load_from_checkpoint(str(model_path))
        self.classifier.to(device)
        self.classifier.eval()
        with open(classmap_path) as fp:
            self.classmap = json.load(fp)

    def __call__(self, bbox_features: torch.Tensor, class_labels: list[str]) -> list[str]:
        """Find the object entity for all detected objects."""
        with torch.no_grad():
            predictions = self.classifier.inference_step({"features": bbox_features})

        entities = []
        for idx, class_label in enumerate(class_labels):
            if class_label in self.classmap["class_labels"]:
                entity_int = predictions[idx]
                entity_str = self.classmap["idx2label"][str(entity_int.item())]

                # TODO: remove this with the new entity classifier
                if entity_str == "Round Table":
                    entity_str = "Table"
                entities.append(entity_str)
            else:
                entities.append(class_label)
        return entities

    def move_to_device(self, device: torch.device) -> None:
        """Moves the mlp to the specified device."""
        self.classifier.to(device)
        self.classifier.eval()


class EntityClassifier(Module):
    """EntiyClassifer MLP."""

    def __init__(
        self,
        in_features: int = 2048,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_classes: int = 10,
    ):
        super().__init__()
        self._in_features = in_features
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self._dropout = dropout
        self.classifier = self.make_layers()

    def make_layers(self) -> Sequential:
        """Make a simple 2 layer MLP."""
        layers = []

        layers.append(Linear(self._in_features, self._hidden_dim))  # type: ignore[arg-type]
        layers.append(BatchNorm1d(self._hidden_dim))  # type: ignore[arg-type]
        layers.append(Dropout(self._dropout))  # type: ignore[arg-type]
        layers.append(Linear(self._hidden_dim, self._num_classes))  # type: ignore[arg-type]

        return Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass."""
        out = self.classifier(batch["features"])
        return {"out": out}


class EntityPolicy(pl.LightningModule):
    """Entity Lightning Module."""

    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()

        self.classifier = EntityClassifier(num_classes=num_classes)

    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference step."""
        output = self.classifier(batch)
        out = output["out"]

        return torch.argmax(out, dim=1)
