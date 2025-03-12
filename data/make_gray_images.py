import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import json

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((name, (xmin, ymin, xmax, ymax)))
    return objects

def generate_grayscale_mask(image_path, xml_path, output_mask_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # 创建800x800的背景
    mask = np.zeros((800, 800), dtype=np.uint8)

    objects = parse_xml(xml_path)

    class_to_gray = {
        'ship': 51,
        'vehicle': 102,
        'storagetank': 153,
        'tenniscourt': 204,
        'airplane': 255
    }

    # 检查是否包含至少一个指定的实例
    contains_instance = any(obj[0] in class_to_gray for obj in objects)
    if not contains_instance:
        return False

    for obj in objects:
        class_name, (xmin, ymin, xmax, ymax) = obj
        if class_name in class_to_gray:
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), class_to_gray[class_name], -1)

    # 下采样到512x512
    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
    
    # 将灰度图转换为三通道
    mask_resized_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(output_mask_path, mask_resized_color)

    # 下采样原图到512x512并保存
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img_resized

def generate_prompt_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    INSTANCES = ['ship', 'vehicle', 'storagetank', 'tenniscourt', 'airplane']
    object_counts = {instance: 0 for instance in INSTANCES}
    
    for obj in root.findall('.//object'):
        name = obj.find('name').text.lower()
        if name in INSTANCES:
            object_counts[name] += 1
    
    if sum(object_counts.values()) == 0:
        return None
    
    prompt = "a remote sensing image with "
    instance_descriptions = []
    for name, count in object_counts.items():
        if count > 0:
            instance_descriptions.append(f"{count} {name}{'s' if count > 1 else ''}")
    
    prompt += ", ".join(instance_descriptions)
    
    return prompt.strip()

def process_all_images(images_folder, annotations_folder, output_mask_folder, output_image_folder, output_json_file):
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    
    image_count = 0
    skipped_count = 0
    json_data = []

    for image_file in os.listdir(images_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, image_file)
            xml_file = os.path.splitext(image_file)[0] + '.xml'
            xml_path = os.path.join(annotations_folder, xml_file)
            
            if os.path.exists(xml_path):
                output_mask_path = os.path.join(output_mask_folder, os.path.splitext(image_file)[0] + '_mask.png')
                output_image_path = os.path.join(output_image_folder, image_file)
                img_resized = generate_grayscale_mask(image_path, xml_path, output_mask_path)
                if img_resized is not False:
                    cv2.imwrite(output_image_path, img_resized)
                    prompt = generate_prompt_from_xml(xml_path)
                    if prompt:
                        json_data.append({
                            "source": os.path.join("data", os.path.relpath(output_mask_path)),
                            "target": os.path.join("data", os.path.relpath(output_image_path)),
                            "prompt": prompt
                        })
                        image_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            else:
                print(f"未找到对应的XML文件: {xml_file}")
                skipped_count += 1
    
    # 写入JSON文件
    with open(output_json_file, 'w', encoding='utf-8') as f:
        for item in json_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"处理完成。共生成 {image_count} 对灰度mask和原图，跳过 {skipped_count} 个图像。")
    print(f"JSON文件已生成: {output_json_file}")

# 使用示例
images_folder = "images"
annotations_folder = "annotations"
output_mask_folder = "grayscale_masks_512"
output_image_folder = "original_images_512"
output_json_file = "image_data.json"

process_all_images(images_folder, annotations_folder, output_mask_folder, output_image_folder, output_json_file)
print("所有灰度mask和原图生成完成，并已生成对应的JSON文件！")
