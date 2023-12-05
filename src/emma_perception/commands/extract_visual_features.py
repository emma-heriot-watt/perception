import argparse

from maskrcnn_benchmark.config import cfg
from pytorch_lightning import Trainer
from scene_graph_benchmark.config import sg_cfg

from emma_perception.callbacks.callbacks import VisualExtractionCacheCallback
from emma_perception.commands.download_checkpoints import (
    download_arena_checkpoint,
    download_vinvl_checkpoint,
)
from emma_perception.datamodules.visual_extraction_dataset import ImageDataset, PredictDataModule
from emma_perception.models.vinvl_extractor import VinVLExtractor, VinVLTransform


def parse_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)  # type: ignore[assignment]
    parser.add_argument(
        "-i",
        "--images_dir",
        required=True,
        help="Path to a folder of images to extract features from",
    )
    parser.add_argument(
        "--is_arena",
        action="store_true",
        help="If we are extracting features from the Arena images, use the Arena checkpoint",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-w", "--num_workers", type=int, default=0)
    parser.add_argument(
        "-c", "--output_dir", default="storage/data/cache", help="Path to store visual features"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for visual feature extraction",
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

    if args.is_arena:
        cfg.MODEL.WEIGHT = download_arena_checkpoint().as_posix()
    else:
        cfg.MODEL.WEIGHT = download_vinvl_checkpoint().as_posix()

    dataset = ImageDataset(
        input_path=args.images_dir, preprocess_transform=VinVLTransform(cfg=cfg)
    )
    dm = PredictDataModule(
        dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    extractor = VinVLExtractor(cfg=cfg)
    trainer = Trainer(
        gpus=args.num_gpus,
        callbacks=[VisualExtractionCacheCallback(cache_dir=args.output_dir, cache_suffix=".pt")],
    )
    trainer.predict(extractor, dm)


if __name__ == "__main__":
    main()
