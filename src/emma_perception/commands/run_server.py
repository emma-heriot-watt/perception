import sys
from io import BytesIO

import torch
import uvicorn
from emma_common.api.instrumentation import instrument_app
from emma_common.aws.cloudwatch import add_cloudwatch_handler_to_logger
from emma_common.logging import (
    InstrumentedInterceptHandler,
    logger,
    setup_logging,
    setup_rich_logging,
)
from fastapi import FastAPI, HTTPException, ORJSONResponse, Response, UploadFile, status
from maskrcnn_benchmark.config import cfg
from opentelemetry import trace
from PIL import Image
from pydantic import BaseModel
from scene_graph_benchmark.config import sg_cfg

from emma_perception._version import __version__  # noqa: WPS436
from emma_perception.api import ApiSettings, ApiStore, extract_features_for_batch, parse_api_args
from emma_perception.models.vinvl_extractor import VinVLExtractor, VinVLTransform


tracer = trace.get_tracer(__name__)

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
    try:
        api_store.extractor.to(torch.device(device.device))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change the model device from {current_device} to {requested_device}".format(
                current_device=api_store.extractor.device, requested_device=device.device
            ),
        )

    return "success"


@app.post("/features", response_model=ORJSONResponse)
async def get_features_for_image(input_file: UploadFile) -> ORJSONResponse:
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
    with tracer.start_as_current_span("Load image"):
        image_bytes = await input_file.read()
        pil_image = Image.open(BytesIO(image_bytes))

    features = extract_features_for_batch(pil_image, api_store, settings.batch_size)

    with tracer.start_as_current_span("Build response"):
        repsonse_content = features[0].dict()

    return ORJSONResponse(repsonse_content)


@app.post("/batch_features", response_class=ORJSONResponse)
async def get_features_for_images(images: list[UploadFile]) -> ORJSONResponse:
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

    with tracer.start_as_current_span("Open images"):
        for input_file in images:
            byte_image = await input_file.read()

            io_image = BytesIO(byte_image)
            io_image.seek(0)
            open_images.append(Image.open(io_image))

    extracted_features = extract_features_for_batch(open_images, api_store, settings.batch_size)

    with tracer.start_as_current_span("Build response"):
        response_content = [feature.dict() for feature in extracted_features]

    return ORJSONResponse(response_content)


def main() -> None:
    """Run the API, exactly the same as the way TEACh does it."""
    if settings.traces_to_opensearch:
        instrument_app(
            app,
            otlp_endpoint=settings.otlp_endpoint,
            service_name=settings.opensearch_service_name,
            service_version=__version__,
            service_namespace="SimBot",
        )
        setup_logging(sys.stdout, InstrumentedInterceptHandler())
    else:
        setup_rich_logging(rich_traceback_show_locals=False)

    if settings.log_to_cloudwatch:
        add_cloudwatch_handler_to_logger(
            boto3_profile_name=settings.aws_profile,
            log_stream_name=settings.watchtower_log_stream_name,
            log_group_name=settings.watchtower_log_group_name,
            send_interval=1,
            enable_trace_logging=settings.traces_to_opensearch,
        )

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=settings.num_workers,
    )


if __name__ == "__main__":
    main()
