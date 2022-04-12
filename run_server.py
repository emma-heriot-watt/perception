import argparse
import io
import json
import random
import string
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from maskrcnn_benchmark.config import cfg
from PIL import Image
from pydantic import BaseModel
from pytorch_lightning import Trainer
from scene_graph_benchmark.config import sg_cfg
from torch.utils.data import Dataset

from src.datamodules.visual_extraction_dataset import DatasetReturn
from src.models.vinvl_extractor import VinVLExtractor, VinVLTransform


app = FastAPI()


class DummyDataset(Dataset[DatasetReturn]):
    """Simple dataset of images.

    All images are assumed to be within a single folder.
    """

    def __init__(self, pil_img: Image, transform: Optional[VinVLTransform] = None) -> None:
        self.dataset = [pil_img]
        self.dataset_size = len(self.dataset)
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> DatasetReturn:
        """Return a sample."""
        img = self.dataset[idx]
        img_size = img.size
        if self.transform is not None:
            img = self.transform(img)
        return DatasetReturn(
            img=img, ids=self._make_sample_id(), width=img_size[0], height=img_size[1]
        )

    def _make_sample_id(self, num_charas: int = 12) -> str:
        return "".join(random.sample(string.ascii_lowercase, num_charas))


class ExtractedFeatures(BaseModel):
    """Base model for returning bbox features."""

    bbox_features: list[list[float]]
    bbox_coords: list[list[float]]
    bbox_probas: list[list[float]]
    cnn_features: list[list[float]]
    class_labels: list[str]


@app.post("/features")
async def get_features(input_file: UploadFile) -> ExtractedFeatures:
    """Endpoint for receiving features for a binary image.

    Example:
        import requests
        url = "http://127.0.0.1:8000/features"
        files = {"input_file": open("cat.jpg", "rb")}
        response = requests.post(url, files=files)
        data = response.json()

    Args:
        input_file (UploadFile): The binary image received from a post request

    Returns:
        features (ExtractedFeatures): A pydantic BaseModel with the features of the bounding boxes
        coordinates, and their probabilities as well as the global cnn features.
    """
    image_bytes = await input_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))

    dataset = DummyDataset(pil_img=pil_image, transform=transform)
    with torch.no_grad():
        cnn_features: list[torch.Tensor] = []
        hook = extractor.extractor.backbone.register_forward_hook(
            lambda module, inp, output: cnn_features.append(output)
        )
        predictions = extractor.extractor(dataset[0]["img"].unsqueeze(0).to(device))
        hook.remove()
        cnn_feats = cnn_features[0][0]  # type: ignore[assignment]
    predictions = predictions[0]
    predictions = predictions.resize((dataset[0]["width"], dataset[0]["height"]))

    bbox_probas = predictions.get_field("scores_all")
    class_labels = bbox_probas.argmax(dim=1).tolist()
    idx_labels = bbox_probas.argmax(dim=1).tolist()
    class_labels = [class_map_dict["idx_to_label"][str(idx)] for idx in idx_labels]

    features = ExtractedFeatures(
        bbox_features=predictions.get_field("box_features").tolist(),
        bbox_coords=predictions.bbox.tolist(),
        bbox_probas=bbox_probas,
        cnn_features=cnn_feats.mean(dim=(-2, -1)).tolist(),  # type: ignore[assignment]
        class_labels=class_labels,
    )
    return features


def parse_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser(prog="PROG")
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--config_file", required=True, metavar="FILE", help="path to VinVL config file"
    )
    parser.add_argument("--classmap_file", required=True, help="Path to VinVL class map file.")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line. Used for VinVL extraction",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host. Defaults to localhost"  # noqa: S104
    )
    parser.add_argument(
        "--port", default=8000, type=int, help="Server port. Defaults to 8000"  # noqa: WPS432
    )
    parser.add_argument(
        "--device_id",
        default=-1,
        type=int,
        help="Device id. Use -1 for cpu and a positive integer to index a gpu.",
    )

    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if torch.cuda.is_available() and args.device_id != -1:
        num_gpus = torch.cuda.device_count()
        if args.device_id >= num_gpus:
            msg = f"You selected {args.device_id} gpu but there are only {num_gpus} available"
            raise OSError(msg)
        device = torch.device(args.device_id)
        cfg.MODEL.DEVICE = args.device_id
    else:
        device = torch.device("cpu")
        cfg.MODEL.DEVICE = "cpu"

    cfg.freeze()
    extractor = VinVLExtractor(cfg=cfg)
    extractor.to(device)
    extractor.eval()
    transform = VinVLTransform(cfg=cfg)
    with open(args.classmap_file) as fp:
        class_map_dict = json.load(fp)
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug", workers=args.num_workers)
