#%%
import cv2
import torch
import torchinfo
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
import pipeline_stable_diffusion_xl_instantid

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

#%%

# Load face encoder
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

face_image = load_image("./examples/yann-lecun_resize.jpg")
face_image = resize_img(face_image)

face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
face_emb = face_info['embedding']
face_kps = pipeline_stable_diffusion_xl_instantid.draw_kps(face_image, face_info['kps'])

#%%

# Path to InstantID models
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'

# Load pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

do_load_controlnet = True

if do_load_controlnet:
    pipe = pipeline_stable_diffusion_xl_instantid.StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
else:
    pipe = pipeline_stable_diffusion_xl_instantid.StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
    )

#%%

pipe.cuda()

#%%

do_load_adapter = True
pipe.load_ip_adapter_instantid(face_adapter, do_load_adapter=do_load_adapter)

#%%

ip_adapter_scale = 0.8
controlnet_conditioning_scale = 0.0
num_inference_steps = 50
seed = 0
# Infer setting
# prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
# n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"

generator = torch.Generator(device="cuda").manual_seed(seed)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image_embeds=face_emb,
    image=face_kps,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    ip_adapter_scale=ip_adapter_scale,
    num_inference_steps=num_inference_steps,
    guidance_scale=5,
    generator=generator
).images[0]

image.save(f'result-{seed=}-{num_inference_steps=}-{do_load_adapter=}-{ip_adapter_scale=}-{controlnet_conditioning_scale=}.jpg')

#%%

torchinfo.summary(pipe.unet)

#%%

from importlib import reload
reload(pipeline_stable_diffusion_xl_instantid)

# %%
