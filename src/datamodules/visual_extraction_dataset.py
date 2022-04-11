import os
from typing import Any, Literal, Optional

import cv2
import torch
from PIL import Image, UnidentifiedImageError
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from src.datamodules.parsers import EpicKitchensParser
from src.utils import get_logger


log = get_logger(__name__)


class DatasetReturn(dict[str, Any]):
    """Dictionary class for accessing samples from ImageDataset and VideoDataset."""

    img: torch.Tensor
    ids: str
    width: int
    height: int


ImageLoaderType = Literal["cv2", "pil"]


class ImageDataset(Dataset[DatasetReturn]):
    """Simple dataset of images."""

    def __init__(
        self,
        input_path: str,
        image_loader: ImageLoaderType = "pil",
        preprocess_transform: Optional[Compose] = None,
    ) -> None:

        self.input_path = input_path
        self.dataset: list[str] = []
        self.image_loader = self._make_loader(image_loader)
        for fname in os.listdir(self.input_path):
            try:
                Image.open(os.path.join(self.input_path, fname))
            except UnidentifiedImageError:
                log.error(f"Could not read {os.path.join(self.input_path, fname)}")
                continue
            self.dataset.append(os.path.join(self.input_path, fname))
        self.dataset_size = len(self.dataset)
        self.transform = preprocess_transform

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> DatasetReturn:
        """Return a sample."""
        fname = self.dataset[idx]
        img = self.image_loader(fname)
        img_size = img.size

        if self.transform is not None:
            img = self.transform(img)
        return DatasetReturn(
            img=img, ids=self._make_sample_id(fname), width=img_size[0], height=img_size[1]
        )

    def _make_loader(self, image_loader: ImageLoaderType) -> Any:
        image_loader_func = None
        if image_loader == "pil":
            image_loader_func = self._pilimage
        elif image_loader == "cv2":
            image_loader_func = self._cv2image
        else:
            raise NotImplementedError(f"Unsupported image loader {image_loader}")
        return image_loader_func

    def _pilimage(self, img_name: str) -> Image:
        return Image.open(img_name).convert("RGB")

    def _cv2image(self, img_name: str) -> Image:
        cv2_img = cv2.imread(img_name)
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def _make_sample_id(self, fname: str) -> str:
        return os.path.basename(fname)


class VideoFrameDataset(Dataset[DatasetReturn]):
    """Simple dataset of video frames."""

    def __init__(
        self,
        input_path: str,
        ann_csv: str,
        ann_type: str,
        image_loader: ImageLoaderType = "pil",
        downsample: int = 0,
        preprocess_transform: Optional[Compose] = None,
    ) -> None:

        self.input_path = input_path
        self.dataset: list[str] = []
        self.image_loader = self._make_loader(image_loader)
        self.parser = self.parse_annotations(ann_csv, ann_type, downsample)

        for (_, dirnames, _) in os.walk(self.input_path):
            for dirname in dirnames:
                dirpath = os.path.join(self.input_path, dirname)
                if self.parser:
                    files = self.parser.get_frames(dirpath)
                else:
                    files = [os.path.join(dirpath, fname) for fname in os.listdir(dirpath)]
                self.append_to_dataset(files)

        self.dataset_size = len(self.dataset)
        self.transform = preprocess_transform

    def append_to_dataset(self, files: list[str]) -> None:
        """Appends a list of files to the dataset."""
        for fname in files:
            try:
                Image.open(fname)
            except UnidentifiedImageError:
                log.error(f"Could not read {os.path.join(self.input_path, fname)}")
                continue
            self.dataset.append(fname)

    def parse_annotations(
        self, ann_csv: str, ann_type: str, downsample: int
    ) -> EpicKitchensParser:
        """Parses the annotations of a video dataset with one of the available parsers."""
        parser = None
        if ann_csv is not None:
            if ann_type == "epic_kitchens":
                parser = EpicKitchensParser(ann_csv, downsample)
            else:
                raise NotImplementedError(f"Unsupported annotation type {ann_type}")
        return parser

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> DatasetReturn:
        """Return a sample."""
        fname = self.dataset[idx]
        img = self.image_loader(fname)
        img_size = img.size
        if self.transform is not None:
            img = self.transform(img)
        return DatasetReturn(
            img=img, ids=self._make_sample_id(fname), width=img_size[0], height=img_size[1]
        )

    def _make_loader(self, image_loader: ImageLoaderType) -> Any:
        image_loader_func = None
        if image_loader == "pil":
            image_loader_func = self._pilimage
        elif image_loader == "cv2":
            image_loader_func = self._cv2image
        else:
            raise NotImplementedError(f"Unsupported image loader {image_loader}")
        return image_loader_func

    def _pilimage(self, img_name: str) -> Image:
        return Image.open(img_name).convert("RGB")

    def _cv2image(self, img_name: str) -> Image:
        cv2_img = cv2.imread(img_name)
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def _make_sample_id(self, fname: str) -> str:
        prefix = os.path.basename(os.path.dirname(fname))
        return f"{prefix}_{os.path.basename(fname)}"


class PredictDataModule(LightningDataModule):
    """A simple data module for predictions."""

    def __init__(self, dataset: Dataset[DatasetReturn], batch_size: int = 42) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset

    def predict_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Defines the dataset to make predictions."""
        dataset_predict = DataLoader(self.dataset, self.batch_size)
        return dataset_predict
