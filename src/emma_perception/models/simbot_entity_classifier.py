from pathlib import Path

import torch
from torch.nn import CosineSimilarity


class SimBotEntityClassifier:
    """SimBotEntityClassifer class."""

    def __init__(self, class_centroids_path: Path) -> None:
        (centroids_vectors, centroids_names) = self._vectorise(class_centroids_path)
        self.centroids_vectors = centroids_vectors
        self.centroids_names = centroids_names
        self._cosine_distance = CosineSimilarity()

    def __call__(self, bbox_features: list[list[float]], class_labels: list[str]) -> list[str]:
        """Find the object entity for all detected objects."""
        entities = []
        bbox_tensor = torch.tensor(bbox_features)
        for idx, class_label in enumerate(class_labels):
            centroid_vectors = self.centroids_vectors.get(class_label, None)
            if centroid_vectors is not None:
                entity_index = self._cosine_distance(bbox_tensor[idx], centroid_vectors).argmax()
                entities.append(self.centroids_names[class_label][entity_index])
            else:
                entities.append(class_label)
        return entities

    def _vectorise(
        self, class_centroids_path: Path
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
        centroids = torch.load(class_centroids_path)
        object_class_centroids_vectors = {}
        object_class_centroids_names = {}
        for object_class, centroid_dict in centroids.items():
            vectors = torch.stack(list(centroid_dict.values()))
            names_list = list(centroid_dict.keys())
            object_class_centroids_vectors[object_class] = vectors
            object_class_centroids_names[object_class] = names_list
        return object_class_centroids_vectors, object_class_centroids_names
