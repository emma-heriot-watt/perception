from pathlib import Path

from emma_common import logger
from emma_common.hf import download_file


HF_REPO_ID = "emma-heriot-watt/models"
VINVL_CHECKPOINT_NAME = "vinvl_pretrained.pth"
ARENA_CHECKPOINT_NAME = "vinvl_finetune_arena.ckpt"


def download_arena_checkpoint(
    *, hf_repo_id: str = HF_REPO_ID, file_name: str = ARENA_CHECKPOINT_NAME
) -> Path:
    """Download the fine-tuned checkpoint on the Alexa Arena."""
    file_path = download_file(repo_id=hf_repo_id, repo_type="model", filename=file_name)
    logger.info(f"Downloaded {file_name}")
    return file_path


def download_vinvl_checkpoint(
    *, hf_repo_id: str = HF_REPO_ID, file_name: str = VINVL_CHECKPOINT_NAME
) -> Path:
    """Download the pre-trained VinVL checkpoint."""
    file_path = download_file(repo_id=hf_repo_id, repo_type="model", filename=file_name)
    logger.info(f"Downloaded {file_name}")
    return file_path
