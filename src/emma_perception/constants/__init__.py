from pathlib import Path


CONSTANTS_DIR_PATH = Path(__file__).parent.resolve()


VINVL_CONFIG_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4.yaml")
VINVL_ALFRED_CONFIG_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_alfred.yaml")
VINVL_SIMBOT_CONFIG_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_simbot_customised.yaml")


VINVL_ALFRED_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_alfred_classmap.json")
VINVL_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_classmap.json")
# VINVL_SIMBOT_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_simbot_classmap.json")
# VINVL_SIMBOT_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath("vinvl_x152c4_simbot_classmap_v124.json")
VINVL_SIMBOT_CLASS_MAP_PATH = CONSTANTS_DIR_PATH.joinpath(
    "vinvl_x152c4_simbot_classmap_v124_customised.json"
)
SIMBOT_ENTITY_CLASSIFER_CENTROID_PATH = CONSTANTS_DIR_PATH.joinpath("entity_centroids.pt")
SIMBOT_ENTITY_MLPCLASSIFIER_PATH = CONSTANTS_DIR_PATH.joinpath("entity_classifier.ckpt")
SIMBOT_ENTITY_MLPCLASSIFIER_CLASSMAP_PATH = CONSTANTS_DIR_PATH.joinpath(
    "entity_classlabel_map.json"
)
