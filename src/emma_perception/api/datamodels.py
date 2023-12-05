import argparse
import json
from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

import torch
from pydantic import BaseSettings

from emma_perception.constants import (
    VINVL_ALFRED_CLASS_MAP_PATH,
    VINVL_CLASS_MAP_PATH,
    VINVL_SIMBOT_CLASS_MAP_PATH,
    VINVL_SIMBOT_CONFIG_PATH,
)
from emma_perception.models.simbot_entity_classifier import SimBotMLPEntityClassifier
from emma_perception.models.vinvl_extractor import VinVLExtractor, VinVLTransform


ClassmapType = Literal["alfred", "original", "simbot"]


class AlfredClassMap(TypedDict):
    """Classmap for class to idx."""

    label_to_idx: dict[str, int]
    idx_to_label: dict[str, str]


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 5500
    host: str = "0.0.0.0"  # noqa: S104
    num_workers: int = 1
    # Use a positive number to index the GPU, or a negative number for the CPU
    device_id: int = -1
    log_level: str = "info"
    # Dictionary containing the class information for the object detector
    classmap_type: ClassmapType = "simbot"
    # batch size used to extract visual features
    batch_size: int = 2

    def object_classmap(self) -> AlfredClassMap:
        """Get the mapping of objects to class indices."""
        if self.classmap_type == "alfred":
            classmap_file = VINVL_ALFRED_CLASS_MAP_PATH
        elif self.classmap_type == "original":
            classmap_file = VINVL_CLASS_MAP_PATH
        elif self.classmap_type == "simbot":
            classmap_file = VINVL_SIMBOT_CLASS_MAP_PATH
        else:
            raise ValueError(f"Invalid classmap type: {self.classmap_type}")

        with open(classmap_file) as in_file:
            return json.load(in_file)


@dataclass(init=False)
class ApiStore:
    """Common state for the API."""

    extractor: VinVLExtractor
    transform: VinVLTransform
    device: torch.device
    entity_classifier: Optional[SimBotMLPEntityClassifier] = None


def parse_api_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        metavar="FILE",
        help="path to VinVL config file",
        default=VINVL_SIMBOT_CONFIG_PATH.as_posix(),
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line. Used for VinVL extraction",
    )
    return parser.parse_args()
