import argparse

from maskrcnn_benchmark.config import cfg
from pytorch_lightning import Trainer
from scene_graph_benchmark.config import sg_cfg

from src.callbacks.callbacks import VisualExtractionCacheCallback
from src.datamodules.visual_extraction_dataset import (
    ImageDataset,
    PredictDataModule,
    VideoFrameDataset,
)
from src.models.vinvl_extractor import VinVLExtractor, VinVLTransform


def parse_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser(prog="PROG")

    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("-i", "--input_path", required=True, help="Path to input dataset")
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-cs", "--cache_suffix", default=".pt", help="Extension of cached files")
    parser.add_argument("--config_file", metavar="FILE", help="path to VinVL config file")
    parser.add_argument("--return_predictions", action="store_true")

    parser.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="Downsampling factor for videos. If 0 then no downsampling is performed",
    )

    parser.add_argument(
        "-c", "--cache", default="storage/data/cache", help="Path to store visual features"
    )

    parser.add_argument(
        "-d", "--dataset", required=True, choices=["images", "frames"], help="Dataset type"
    )
    parser.add_argument(
        "-a",
        "--ann_csv",
        help="Path to annotation csv file. Used for video datasets to select only the frames that have annotations",
    )

    parser.add_argument(
        "-at",
        "--ann_type",
        choices=["epic_kitchens"],
        default="epic_kitchens",
        help="Annotation parser for video datasets",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line. Used for VinVL extraction",
    )

    return parser.parse_args()


def main() -> None:
    """Main function for visual feature extraction."""
    args = parse_args()

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    extractor = VinVLExtractor(cfg=cfg)
    transform = VinVLTransform(cfg=cfg)

    dataset = None
    if args.dataset == "images":
        dataset = ImageDataset(input_path=args.input_path, preprocess_transform=transform)
    elif args.dataset == "frames":
        dataset = VideoFrameDataset(
            input_path=args.input_path,
            ann_csv=args.ann_csv,
            ann_type=args.ann_type,
            preprocess_transform=transform,
            downsample=args.downsample,
        )
    else:
        raise OSError(f"Unsupported dataset type {args.dataset}")

    dm = PredictDataModule(dataset=dataset, batch_size=args.batch_size)
    trainer = Trainer(
        gpus=args.gpus,
        callbacks=[
            VisualExtractionCacheCallback(cache_dir=args.cache, cache_suffix=args.cache_suffix)
        ],
    )
    trainer.predict(extractor, dm, return_predictions=args.return_predictions)


if __name__ == "__main__":
    main()
