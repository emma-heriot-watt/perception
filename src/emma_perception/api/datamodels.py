import argparse
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseSettings

from emma_perception.models.vinvl_extractor import VinVLExtractor, VinVLTransform


ClassmapType = Literal["alfred", "original", "simbot"]


class ApiSettings(BaseSettings):
    """Common settings, which can also be got from the environment vars."""

    port: int = 5500
    host: str = "0.0.0.0"  # noqa: S104
    num_workers: int = 1
    # Use a positive number to index the GPU, or a negative number for the CPU
    device_id: int = -1
    log_level: str = "info"
    # Dictionary containing the class information for the object detector
    classmap_type: ClassmapType = "alfred"


@dataclass(init=False)
class ApiStore:
    """Common state for the API."""

    extractor: VinVLExtractor
    transform: VinVLTransform
    device: torch.device


def parse_api_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", required=True, metavar="FILE", help="path to VinVL config file"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line. Used for VinVL extraction",
    )
    return parser.parse_args()
