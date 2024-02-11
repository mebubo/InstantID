import os
import gdown
import zipfile
from huggingface_hub import hf_hub_download

os.makedirs("models", exist_ok=True)
gdown.download(id="18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8", output="models/antelopev2.zip")
with zipfile.ZipFile('models/antelopev2.zip','r') as zip_ref:
  zip_ref.extractall('models')

hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")
