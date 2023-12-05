import argparse
from pathlib import Path

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from scene_graph_benchmark.config import sg_cfg
from torch.utils.data import DataLoader


def prepare_configs(args: argparse.Namespace) -> None:
    """Prepare the config."""
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


def make_output_folders(
    dataset_names: list[str],
    model_name: str,
    output_dir: Path,
) -> list[Path]:
    """Prepare the list of output folders."""
    output_folders = []
    if len(dataset_names) == 1:
        output_folder = output_dir.joinpath("inference", model_name)
        mkdir(output_folder)
        output_folders = [output_folder]
    else:
        for dataset_name in dataset_names:
            dataset_name1 = dataset_name.replace("/", "_")
            output_folder = output_dir.joinpath("inference", dataset_name1, model_name)

            mkdir(output_folder)
            output_folders.append(output_folder)
    return output_folders


def post_process_outputs(dataloader: DataLoader, output_folder: Path) -> None:  # type: ignore[type-arg]
    """Post process predictions from a single file to one file per image."""
    id_to_img_map = dataloader.dataset.id_to_img_map
    predictions = torch.load(output_folder.joinpath("predictions.pth"))
    features_path = output_folder.joinpath("image_features_forced_bboxes")
    Path.mkdir(features_path, exist_ok=True)
    for idx, pred in predictions.items():
        image_info = dataloader.dataset.get_img_info(idx)
        width = image_info["width"]
        height = image_info["height"]
        pred = pred.resize((width, height))
        feature_dict = {
            "bbox_features": pred.get_field("box_features"),
            "bbox_coords": pred.bbox,
            "bbox_probas": pred.get_field("scores_all"),
            "cnn_features": pred.get_field("cnn_features"),
            "width": width,
            "height": height,
        }
        features_file_name = f"{str(id_to_img_map[idx]).zfill(12)}.pt"  # noqa: WPS432
        torch.save(feature_dict, features_path.joinpath(features_file_name))
