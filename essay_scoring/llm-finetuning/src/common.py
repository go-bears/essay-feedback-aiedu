import os
from pathlib import PurePosixPath
from typing import Union

import modal

APP_NAME = "ft-gemma-3"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "false").lower() == "true"

LLAMA_CPP_RELEASE = "b4568"
MINUTES = 60

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

axolotl_image = (
    # modal.Image.from_registry(f"axolotlai/axolotl:main-latest")
    # modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev"
    )
    .pip_install(
        "huggingface_hub[hf_transfer]==0.30.1",
        "hf-transfer==0.1.5",
        "fastapi==0.110.0",
        "pydantic",
        "transformers==4.51.0",
        "unsloth"
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
    .entrypoint([])
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install("vllm==0.8.2", "torch==2.6.0", "transformers==4.50.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    .env({"VLLM_USE_V1": "0"})
    .entrypoint([])
)

app = modal.App(
    APP_NAME,
    secrets=[
        modal.Secret.from_name("huggingface-rw-joaquin"),
        modal.Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
        *([modal.Secret.from_name("wandb")] if ALLOW_WANDB else []),
    ],
)

# Volumes for pre-trained models and training runs.
pretrained_volume = modal.Volume.from_name(
    "example-pretrained-vol", create_if_missing=True
)
runs_volume = modal.Volume.from_name("example-runs-vol", create_if_missing=True)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
