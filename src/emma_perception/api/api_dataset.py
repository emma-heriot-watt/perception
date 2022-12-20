import random
import string
from typing import Optional, Union

from PIL.Image import Image
from torch.utils.data import Dataset

from emma_perception.api.instrumentation import get_tracer
from emma_perception.datamodules.visual_extraction_dataset import DatasetReturn
from emma_perception.models.vinvl_extractor import VinVLTransform


tracer = get_tracer(__name__)


class ApiDataset(Dataset[DatasetReturn]):
    """Simple dataset of images from an API call."""

    def __init__(
        self, images: Union[Image, list[Image]], transform: Optional[VinVLTransform] = None
    ) -> None:
        if not isinstance(images, list):
            images = [images]

        self.dataset = images
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
            with tracer.start_as_current_span("Transform image"):
                img = self.transform(img)

        return DatasetReturn(
            img=img, ids=self._make_sample_id(), width=img_size[0], height=img_size[1]
        )

    def _make_sample_id(self, num_charas: int = 12) -> str:
        return "".join(random.sample(string.ascii_lowercase, num_charas))
