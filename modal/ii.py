import io
from pathlib import Path

from modal import Image, Mount, Stub, asgi_app, build, enter, gpu, method

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "diffusers",
        "insightface",
        "onnxruntime",
        "accelerate",
        "transformers",
        "gdown"
    )
)

stub = Stub("instant-id")

with image.imports():
    from huggingface_hub import hf_hub_download, snapshot_download
    import gdown
    import os
    import zipfile

    import cv2
    import torch
    import numpy as np
    from PIL import Image

    from diffusers.utils import load_image
    from diffusers.models import ControlNetModel

    from insightface.app import FaceAnalysis
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240, image=image, 
          mounts=[Mount.from_local_dir("./examples", remote_path="/examples")]
        )
class Model:

    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    controlnet_path = f'./checkpoints/ControlNetModel'


    @build()
    def build(self):
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
        gdown.download(url="https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing", output="./models/", quiet=False, fuzzy=True)
        with zipfile.ZipFile('models/antelopev2.zip','r') as zip_ref:
            zip_ref.extractall('models')

        # snapshot_download(Model.base_model_path)

        controlnet = ControlNetModel.from_pretrained(Model.controlnet_path, torch_dtype=torch.float16)
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            Model.base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

    @enter()
    def enter(self):
        self.app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Path to InstantID models
        face_adapter = f'./checkpoints/ip-adapter.bin'

        # Load pipeline
        controlnet = ControlNetModel.from_pretrained(Model.controlnet_path, torch_dtype=torch.float16)

        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            Model.base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        self.pipe.cuda()
        self.pipe.load_ip_adapter_instantid(face_adapter)

    @method()
    def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        face_image = load_image("/examples/yann-lecun_resize.jpg")
        face_image = resize_img(face_image)

        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])

        prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
        n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

        image = self.pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.8,
            ip_adapter_scale=0.8,
            num_inference_steps=30,
            guidance_scale=5,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

@stub.local_entrypoint()
def main(prompt: str):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("./stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)

