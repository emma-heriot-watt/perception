import argparse
import os
from pathlib import Path
from typing import Any

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize
from scene_graph_benchmark.scene_parser import SceneParser

from emma_perception.models import AttrRCNNGlobalFeatures
from emma_perception.utils import get_logger
from emma_perception.utils.forced_bbox_extraction import (
    make_output_folders,
    post_process_outputs,
    prepare_configs,
)


logger = get_logger(__name__)


def run_test(config: cfg, model: Any, distributed: bool, model_name: str) -> None:
    """Source: https://github.com/microsoft/scene_graph_benchmark/blob/main/tools/train_sg_net.py."""
    if distributed and model.hasattr("module"):
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    dataset_names = config.DATASETS.TEST
    if not config.OUTPUT_DIR:
        raise AssertionError("OUTPUT_DIR was not provided.")
    output_folders = make_output_folders(
        dataset_names=dataset_names,
        model_name=model_name,
        output_dir=Path(
            config.OUTPUT_DIR,
        ),
    )
    data_loaders_val = make_data_loader(config, is_train=False, is_distributed=distributed)
    labelmap_file = config_dataset_file(config.DATA_DIR, config.DATASETS.LABELMAP_FILE)
    for output_folder, dataset_name, data_loader_val in zip(  # noqa: WPS352
        output_folders, dataset_names, data_loaders_val
    ):
        inference(
            model,
            config,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if config.MODEL.RETINANET_ON else config.MODEL.RPN_ONLY,
            bbox_aug=config.TEST.BBOX_AUG.ENABLED,
            device=config.MODEL.DEVICE,
            expected_results=config.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=config.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            skip_performance_eval=config.TEST.SKIP_PERFORMANCE_EVAL,
            labelmap_file=labelmap_file,
            save_predictions=config.TEST.SAVE_PREDICTIONS,
        )

        synchronize()
        post_process_outputs(data_loader_val, output_folder)


def main() -> None:
    """Prepare the model and config."""
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    prepare_configs(args)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=cfg.DISTRIBUTED_BACKEND, init_method="env://")
        synchronize()

    logger.info(f"Using {num_gpus} GPUs")
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)\n")
    logger.info(collect_env_info())

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNNGlobalFeatures(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    checkpointer.load(ckpt, use_latest=args.ckpt is None)
    model_name = Path(ckpt).stem

    run_test(cfg, model, args.distributed, model_name)


if __name__ == "__main__":
    main()
