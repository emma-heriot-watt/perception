from pathlib import Path

import torch
from torch.nn import CosineSimilarity


class SimBotEntityClassifier:
    """SimBotEntityClassifer class."""

    default_tensor_device: torch.device = torch.device("cpu")

    def __init__(
        self, class_centroids_path: Path, device: torch.device = default_tensor_device
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
