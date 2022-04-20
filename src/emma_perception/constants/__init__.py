import json
from pathlib import Path
from typing import TypedDict


CONSTANTS_DIR_PATH = Path(__file__).parent.resolve()


VINVL_CONFIG_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4.yaml")
VINVL_ALFRED_CONFIG_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_alfred.yaml")


class AlfredClassMap(TypedDict):
    label_to_idx: dict[str, int]
    idx_to_label: dict[str, str]


VINVL_ALFRED_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152_c4_alfred_classmap.json")
VINVL_ALFRED_CLASS_MAP: AlfredClassMap = json.loads(VINVL_ALFRED_CLASS_MAP_PATH.read_bytes())
