from huggingface_hub import hf_hub_download
import gdown
import os
import zipfile

# download models
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="./checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="./checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints"
)
hf_hub_download(
    repo_id="latent-consistency/lcm-lora-sdxl",
    filename="pytorch_lora_weights.safetensors",
    local_dir="./checkpoints",
)

destination_file = './models/antelopev2.zip'
if not os.path.exists(destination_file):
    gdown.download(url="https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing", output=destination_file, quiet=False, fuzzy=True)
    with zipfile.ZipFile(destination_file, 'r') as zip_ref:
        zip_ref.extractall('models')

base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
controlnet_path = f'./checkpoints/ControlNetModel'

import torch
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from diffusers.models import ControlNetModel

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
)