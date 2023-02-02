import torch
from codetiming import Timer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg

from emma_perception.models.vinvl_extractor import VinVLExtractor


# ------------------------------ Load the model ------------------------------ #
cfg.set_new_allowed(True)
cfg.merge_from_other_cfg(sg_cfg)
cfg.merge_from_file("src/emma_perception/constants/vinvl_x152c4_simbot_customised.yaml")
opts = [
    "MODEL.WEIGHT",
    "storage/models/vinvl_vg_x152c4_simbot_v124_customised.pth",
    "MODEL.ROI_HEADS.NMS_FILTER",
    "1",
    "MODEL.ROI_HEADS.SCORE_THRESH",
    "0.2",
    "TEST.IGNORE_BOX_REGRESSION",
    "False",
]
cfg.merge_from_list(opts)
cfg.MODEL.DEVICE = "cuda:0"

NUM_IMAGES = 1
IMAGE_SIZE = 300
image = torch.randn(NUM_IMAGES, 3, IMAGE_SIZE, IMAGE_SIZE)


@Timer(name="non_compiled")
def get_results_from_non_compiled_model(image_tensor):
    batch = {
        "ids": torch.tensor(list(range(1, NUM_IMAGES + 1))),
        "img": image_tensor.detach().clone(),
        "width": torch.tensor([IMAGE_SIZE]),
        "height": torch.tensor([IMAGE_SIZE]),
    }
    extractor = VinVLExtractor(cfg=cfg)
    device = "cuda:0"
    extractor.to(device)
    extractor.eval()
    with torch.inference_mode():
        model_output = extractor.forward(batch)

    return model_output


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


@Timer(name="onnx")
def get_results_from_onnx(image_tensor):
    import onnxruntime as ort

    batch = {
        "0": to_numpy(torch.tensor([1, 2])),
        "1": to_numpy(image_tensor.detach().clone()),
        "2": to_numpy(torch.tensor([IMAGE_SIZE])),
        "3": to_numpy(torch.tensor([IMAGE_SIZE])),
    }

    ort_session = ort.InferenceSession(
        "storage/model/vinvl.onnx", providers=["CUDAExecutionProvider"]
    )
    outputs = ort_session.run(None, batch)
    return outputs


def compile_model(image_tensor):
    extractor = VinVLExtractor(cfg=cfg)
    device = "cuda:0"
    extractor.to(device)
    extractor.eval()
    batch = {
        "ids": torch.tensor([1, 2]),
        "img": image_tensor.detach().clone(),
        "width": torch.tensor([IMAGE_SIZE]),
        "height": torch.tensor([IMAGE_SIZE]),
    }
    torch.onnx.export(extractor, batch, "storage/model/vinvl.onnx", opset_version=11)


def check_model():
    import onnx

    model = onnx.load("storage/model/vinvl.onnx")
    onnx.checker.check_model(model)
    onnx.checker.check_model(model, full_check=True)


compile_model(image)
check_model()

non_compiled_output = get_results_from_non_compiled_model(image)
onnx_output = get_results_from_onnx(image)

print("end")
