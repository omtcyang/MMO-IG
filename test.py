from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# 加载模型
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('checkpoints/epoch=40_ori_dior.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


# 处理函数
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        
        H, W, C = input_image.shape

        # 检测输入图像
        detected_map = input_image
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # 条件设置
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # 设置控制强度
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        
        # 采样
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                  unconditional_guidance_scale=scale,
                                               unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shiis_diffusing=False

        # 解码并处理输出图像
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return  results


# 推理函数
def infer(input_image_path, prompt, a_prompt, n_prompt, num_samples=1, image_resolution=512, ddim_steps=20, guess_mode=False, strength=1.0, scale=9.0, seed=-1, eta=0.0):
    input_image = cv2.imread(input_image_path)  # 从路径读取图像
    results = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
    
    # 保存或返回结3
    for i, result in enumerate(results):
        output_path = f"remove1_output_{i}.png"
        cv2.imwrite(output_path, result)
        print(f"Result saved to {output_path}")

# high quality,exact quantity
# blur,instance overlap
# 调用推理
infer('generated_images2/group_1_remove_1_R3.png', prompt="a remote sensing image with 2 airplanes, 2 vehicles. ", a_prompt="high quality,exact quantity", 
      n_prompt="blur,instance overlap", num_samples=1, image_resolution=512, 
      ddim_steps=20, guess_mode=False, strength=1.0, scale=2.7, seed=-1, eta=0.0)
