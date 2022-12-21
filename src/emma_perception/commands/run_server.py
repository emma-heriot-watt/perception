from argparse import Namespace
from io import BytesIO

import torch
from fastapi import FastAPI, Response, UploadFile, status
from maskrcnn_benchmark.config import cfg
from PIL import Image
from ray import get_gpu_ids, serve
from scene_graph_benchmark.config import sg_cfg

from emma_common.logging import logger
from emma_perception.api import ApiSettings, ApiStore, extract_features_for_batch, parse_api_args
from emma_perception.datamodels import ExtractedFeaturesAPI
from emma_perception.models.vinvl_extractor import VinVLExtractor, VinVLTransform


settings = ApiSettings()
api_store = ApiStore()
app = FastAPI()
logger.info("Initializing EMMA Perception API")


@serve.deployment(
    route_prefix="/",
    num_replicas=settings.num_replicas,
    ray_actor_options={"num_gpus": settings.num_gpus},
)
@serve.ingress(app)
class PerceptionDeployment:
    """Deployment of the perception service."""

    def __init__(self, args: Namespace) -> None:
        """Initialises the perception model."""
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)

        available_devices = get_gpu_ids()
        if available_devices and settings.device_id != -1:
            num_gpus = len(available_devices)

            if settings.device_id >= num_gpus:
                msg = f"You selected {settings.device_id} gpu but there are only {num_gpus} available"
                raise OSError(msg)

            device = torch.device(settings.device_id)
            cfg.MODEL.DEVICE = settings.device_id
        else:
            device = torch.device("cpu")
            cfg.MODEL.DEVICE = "cpu"

        cfg.freeze()
        logger.info(f"Config:{cfg}")

        logger.info(f"Setting up VinVL Extractor on device: {device}")
        extractor = VinVLExtractor(cfg=cfg)
        self.device = device
        extractor.to(self.device)
        extractor.eval()

        logger.info("Setting up the VinVL transform")
        transform = VinVLTransform(cfg=cfg)

        self.extractor = extractor
        self.transform = transform

        logger.info("Setup complete!")

    @app.post("/features", response_model=ExtractedFeaturesAPI)
    async def get_features_for_image(self, input_file: UploadFile) -> ExtractedFeaturesAPI:
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
            features (ExtractedFeaturesAPI): A pydantic BaseModel with the features of the bounding boxes
            coordinates, and their probabilities as well as the global cnn features.
        """
        image_bytes = await input_file.read()
        pil_image = Image.open(BytesIO(image_bytes))

        features = extract_features_for_batch(
            pil_image, self.extractor, self.transform, self.device
        )

        return features[0]

    @app.post("/batch_features")
    async def get_features_for_images(
        self, images: list[UploadFile]
    ) -> list[ExtractedFeaturesAPI]:
        """Endpoint for receiving features for a batch of binary images.

        Example:
            import requests
            url = "http://127.0.0.1:8000/batch_features"
            files = [
                ("images", ("file1", open("storage/data/alfred/000000000.jpg", "rb"))),
                ("images", ("file2", open("storage/data/alfred/000000001.jpg", "rb")))
            ]
            response = requests.post(url, files=files)
            data = response.json()

        Args:
            images (list[UploadFile]): A batch of binary images received from a post request. Assumes they all have the same size.

        Returns:
            features (ExtractedFeaturesAPI): A pydantic BaseModel with the features of the bounding boxes
            coordinates, and their probabilities as well as the global cnn features.
        """
        open_images = []

        for input_file in images:
            byte_image = await input_file.read()

            io_image = BytesIO(byte_image)
            io_image.seek(0)
            open_images.append(Image.open(io_image))

        return extract_features_for_batch(open_images, self.extractor, self.transform, self.device)


@app.get("/")
@app.get("/ping")
@app.get("/healthcheck")
async def root(response: Response) -> str:
    """Ping the API to make sure it is responding."""
    response.status_code = status.HTTP_200_OK
    return "success"


def wait_for_requests() -> None:
    """Uses an infinite loop to wait for incoming requests.

    Without this the Ray node will be shutdown immediately after the main ends. The infinite loop
    prevents this and make sure that we keep listening for incoming requests.
    """
    while True:  # noqa: WPS328, WPS457
        pass  # noqa: WPS420


def main() -> None:
    """Run the API, exactly the same as the way TEACh does it."""
    args = parse_api_args()

    perception = PerceptionDeployment.bind(args)  # type: ignore[attr-defined]
    serve.run(perception, host=settings.host, port=settings.port)

    logger.info("Server initialised.")
    try:
        wait_for_requests()
    except Exception:
        logger.info("Server stopped.")
        serve.shutdown()


if __name__ == "__main__":
    main()
