import json
from pathlib import Path
from typing import TypedDict

from emma_perception.api.datamodels import ApiSettings, ClassmapType


CONSTANTS_DIR_PATH = Path(__file__).parent.resolve()


VINVL_CONFIG_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4.yaml")
VINVL_ALFRED_CONFIG_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_alfred.yaml")


class AlfredClassMap(TypedDict):
    label_to_idx: dict[str, int]
    idx_to_label: dict[str, str]


VINVL_ALFRED_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_alfred_classmap.json")
VINVL_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_classmap.json")
VINVL_SIMBOT_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_simbot_classmap.json")


def _classmap(classmap_type: ClassmapType) -> AlfredClassMap:
    # Returns the map that will be used by the object detector to determine the object class.
    if classmap_type == "alfred":
        classmap_file = VINVL_ALFRED_CLASS_MAP_PATH
    elif classmap_type == "original":
        classmap_file = VINVL_CLASS_MAP_PATH
    elif classmap_type == "simbot":
        classmap_file = VINVL_SIMBOT_CLASS_MAP_PATH
    else:
        raise ValueError(f"Invalid classmap type: {classmap_type}")

    with open(classmap_file) as in_file:
        return json.load(in_file)


settings = ApiSettings()

OBJECT_CLASSMAP = _classmap(settings.classmap_type)
