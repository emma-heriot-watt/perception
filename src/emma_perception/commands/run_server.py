from io import BytesIO
from pathlib import Path

import torch
import uvicorn
from emma_common.api.response import TorchResponse
from emma_common.logging import logger, setup_rich_logging
from fastapi import FastAPI, HTTPException, Response, UploadFile, status
from maskrcnn_benchmark.config import cfg
from PIL import Image
from pydantic import BaseModel
from scene_graph_benchmark.config import sg_cfg

from emma_perception.api import ApiSettings, ApiStore, extract_features_for_batch, parse_api_args
from emma_perception.commands.download_checkpoints import download_arena_checkpoint
from emma_perception.constants import (
    SIMBOT_ENTITY_MLPCLASSIFIER_CLASSMAP_PATH,
    SIMBOT_ENTITY_MLPCLASSIFIER_PATH,
)
from emma_perception.models.simbot_entity_classifier import SimBotMLPEntityClassifier
from emma_perception.models.vinvl_extractor import VinVLExtractor, VinVLTransform


settings = ApiSettings()
api_store = ApiStore()
app = FastAPI()


@app.on_event("startup")
async def startup_event() -> None:
    """Run specific functions when starting up the API."""
    args = parse_api_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    model_path = Path(cfg.MODEL.WEIGHT)
    if not model_path.exists():
        cfg.MODEL.WEIGHT = download_arena_checkpoint().as_posix()

    if torch.cuda.is_available() and settings.device_id != -1:
        num_gpus = torch.cuda.device_count()

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

    api_store.device = device

    logger.info(f"Setting up VinVL Extractor on device: {device}")
    extractor = VinVLExtractor(cfg=cfg)
    extractor.to(device)
    extractor.eval()

    logger.info("Setting up the VinVL transform")
    transform = VinVLTransform(cfg=cfg)

    api_store.extractor = extractor
    api_store.transform = transform
    if settings.classmap_type == "simbot":
        api_store.entity_classifier = SimBotMLPEntityClassifier(
            model_path=SIMBOT_ENTITY_MLPCLASSIFIER_PATH,
            classmap_path=SIMBOT_ENTITY_MLPCLASSIFIER_CLASSMAP_PATH,
            device=device,
        )

    logger.info("Setup complete!")


@app.get("/")
@app.get("/ping")
@app.get("/test")
async def root(response: Response) -> str:
    """Ping the API to make sure it is responding."""
    response.status_code = status.HTTP_200_OK
    return "success"


class DeviceRequestBody(BaseModel):
    """Pydantic model for the request body when updating the model device.

    This is used because the device is passed as a request body.
    """

    device: str


@app.post("/update_model_device", status_code=status.HTTP_200_OK)
async def update_model_device(device: DeviceRequestBody) -> str:
    """Update the device used by the model."""
    new_device = torch.device(device.device)
    try:
        api_store.extractor.to(new_device)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change the model device from {current_device} to {requested_device}".format(
                current_device=api_store.extractor.device, requested_device=device.device
            ),
        ) from err

    if api_store.entity_classifier is not None:
        try:
            api_store.entity_classifier.move_to_device(new_device)
        except Exception as other_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to change the entity classifier device from {current_device} to {requested_device}".format(
                    current_device=api_store.entity_classifier.device,
                    requested_device=device.device,
                ),
            ) from other_err

    return "success"


@app.post("/features", response_class=TorchResponse)
def get_features_for_image(input_file: UploadFile) -> TorchResponse:
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
    logger.debug("Load image")
    image_bytes = input_file.file.read()
    pil_image = Image.open(BytesIO(image_bytes))

    features = extract_features_for_batch(pil_image, api_store, settings.batch_size)

    logger.debug("Build response")
    response = TorchResponse(features[0])

    return response


@app.post("/batch_features", response_class=TorchResponse)
def get_features_for_images(images: list[UploadFile]) -> TorchResponse:
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
    open_images: list[Image.Image] = []

    logger.debug("Load images")
    for input_file in images:
        byte_image = input_file.file.read()
        io_image = BytesIO(byte_image)
        io_image.seek(0)
        open_images.append(Image.open(io_image))

    extracted_features = extract_features_for_batch(open_images, api_store, settings.batch_size)

    logger.debug("Build response")
    response = TorchResponse(extracted_features)

    return response


def main() -> None:
    """Run the API, exactly the same as the way TEACh does it."""
    setup_rich_logging(rich_traceback_show_locals=False)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=settings.num_workers,
    )


if __name__ == "__main__":
    main()
