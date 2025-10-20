import json
import cv2
import einops
import numpy as np
import torch
import random
import os
import argparse
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config


def load_json_records(json_path):
    """加载JSON文件并返回记录列表"""
    with open(json_path, 'r') as file:
        records = [json.loads(line) for line in file]
    return records

def calculate_required_controls(total_images, num_samples):
    """计算所需的控制图像数量"""
    return (total_images + num_samples - 1) // num_samples  # 向上取整

def select_control_images(json_records, required_controls):
    """从JSON记录中选择所需的控制图像及对应的信息"""
    if required_controls > len(json_records):
        raise ValueError(f"JSON文件中的记录不足，需 {required_controls} 条记录，但仅有 {len(json_records)} 条。")
    return json_records[:required_controls]

def process_inference(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
                      ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler):
    # (这个函数内部逻辑是正确的，无需修改)
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], 
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], 
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, 
                                                     verbose=False, eta=eta, 
                                                     unconditional_guidance_scale=scale, 
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # x_samples 输出为 RGB 格式
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results

def main():
    parser = argparse.ArgumentParser(description='使用CLDM模型的推理脚本')
    
    parser.add_argument('--json_file', type=str, required=True, help='包含控制图像及提示信息的JSON文件')
    parser.add_argument('--a_prompt', type=str, default='Accurate number of instances, reasonable instance size, well-distributed', help='Positive prompt description')
    parser.add_argument('--n_prompt', type=str, default='Incorrect number of instances, low resolution, blurry image', help='Negative prompt description')
    parser.add_argument('--num_samples', type=int, default=6, help='Number of samples to generate per control image')
    parser.add_argument('--image_resolution', type=int, default=512, help='Image resolution to process')
    parser.add_argument('--ddim_steps', type=int, default=20, help='Number of DDIM steps for sampling')
    parser.add_argument('--guess_mode', action='store_true', help='Enable guess mode for control scales')
    parser.add_argument('--strength', type=float, default=1.0, help='Control strength')
    parser.add_argument('--scale', type=float, default=3.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed for reproducibility')
    parser.add_argument('--eta', type=float, default=0.0, help='Eta value for DDIM sampling')
    parser.add_argument('--total_images', type=int, default=20000, help='Total number of images to generate')
    parser.add_argument('--output_dir', type=str, default='./generated_wavelet', help='Directory to save generated images')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the model weights file')

    args = parser.parse_args()

    # 加载模型
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(args.weights_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    # 加载JSON文件
    json_records = load_json_records(args.json_file)

    # 计算所需控制图像数量
    required_controls = calculate_required_controls(args.total_images, args.num_samples)

    # 获取所需的控制图像及提示信息
    selected_controls = select_control_images(json_records, required_controls)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    total_generated = 0
    for control in selected_controls:
        try:
            input_image_path = control['source']
            prompt = control['prompt']

            # ：cv2.imread 读取为 BGR 格式
            input_image_bgr = cv2.imread(input_image_path)

            # ：增加对读取失败的检查
            if input_image_bgr is None:
                print(f"警告：无法读取图像 {input_image_path}，跳过此记录。")
                continue

            # 将 BGR 转换为 RGB 以匹配模型输入
            input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)

            # 将 RGB 图像传入
            results_rgb = process_inference(input_image_rgb, prompt, args.a_prompt, args.n_prompt, args.num_samples, 
                                            args.image_resolution, args.ddim_steps, args.guess_mode, 
                                            args.strength, args.scale, args.seed, args.eta, model, ddim_sampler)

            # ：拿到的 'result_rgb' 是 RGB 格式
            for idx, result_rgb in enumerate(results_rgb):
                if total_generated >= args.total_images:
                    break
                
                output_path = os.path.join(args.output_dir, f'output_image_{total_generated}.png')
                
                #：cv2.imwrite 需要 BGR 格式，因此保存前将 RGB 转回 BGR
                result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(output_path, result_bgr)
                print(f'已保存结果到 {output_path}')
                total_generated += 1

        except Exception as e:
            print(f"处理图片时发生错误: {str(e)}")
            print(f"跳过此记录并继续处理下一条。")
            continue
        
        #  增加一个总数检查，如果已生成足够图片，可提前退出外层循环
        if total_generated >= args.total_images:
            print(f"已生成目标数量 {args.total_images} 张图片，任务完成。")
            break

    print(f"总共生成了 {total_generated} 张图片。")

if __name__ == "__main__":
    main()