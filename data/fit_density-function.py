import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        category = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((category, (xmin, ymin, xmax, ymax)))
    return objects

def analyze_images(annotations_folder):
    instance_counts = []
    category_data = {cat: {'centers': [], 'aspect_ratios': [], 'lengths': []} for cat in ['ship', 'vehicle', 'storagetank', 'tenniscourt', 'airplane']}
    category_presence = {cat: [] for cat in category_data.keys()}
    category_counts = {cat: 0 for cat in category_data.keys()}
    
    for xml_file in os.listdir(annotations_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_folder, xml_file)
            objects = parse_xml(xml_path)
            
            image_instances = []
            present_categories = set()
            for obj in objects:
                category, (xmin, ymin, xmax, ymax) = obj
                if category in category_data:
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    width = xmax - xmin
                    height = ymax - ymin
                    if height != 0:
                        aspect_ratio = width / height
                        length = max(width, height)
                        
                        category_data[category]['centers'].append([cx, cy])
                        category_data[category]['aspect_ratios'].append(aspect_ratio)
                        category_data[category]['lengths'].append(length)
                    
                    image_instances.append(category)
                    present_categories.add(category)
                    category_counts[category] += 1
            
            count = len(image_instances)
            if 0 < count <= 30:
                instance_counts.append(count)
            
            for cat in category_data.keys():
                category_presence[cat].append(1 if cat in present_categories else 0)
    
    return category_data, instance_counts, category_presence, category_counts

def fit_density_functions(category_data, instance_counts):
    density_functions = {}
    aspect_ratio_length_models = {}
    
    # 拟合实例数量的密度函数
    kde_instance_counts = stats.gaussian_kde(instance_counts)
    density_functions['instance_counts'] = kde_instance_counts
    
    for category, data in category_data.items():
        centers = np.array(data['centers'])
        aspect_ratios = np.array(data['aspect_ratios'])
        lengths = np.array(data['lengths'])
        
        # 拟合中心点的二维密度函数
        kde_centers = stats.gaussian_kde(centers.T)
        density_functions[f'{category}_centers'] = kde_centers
        
        # 拟合长宽比的密度函数
        kde_aspect_ratios = stats.gaussian_kde(aspect_ratios)
        density_functions[f'{category}_aspect_ratios'] = kde_aspect_ratios
        
        # 拟合长度的密度函数
        kde_lengths = stats.gaussian_kde(lengths)
        density_functions[f'{category}_lengths'] = kde_lengths
        
        # 拟合长宽比和长度之间的线性关系
        model = LinearRegression()
        model.fit(aspect_ratios.reshape(-1, 1), lengths)
        aspect_ratio_length_models[category] = model
    
    return density_functions, aspect_ratio_length_models

def plot_density_functions(category_data, density_functions, aspect_ratio_length_models, instance_counts, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    sns.set_theme(style="whitegrid")
    
    # 绘制实例数量密度函数
    kde_instance_counts = density_functions['instance_counts']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(instance_counts, kde=True, ax=ax)
    ax.set_title('Instance Count Density')
    ax.set_xlabel('Instance Count')
    ax.set_ylabel('Density')
    plt.savefig(os.path.join(output_folder, 'instance_count_density.png'))
    plt.close()
    
    for category, data in category_data.items():
        centers = np.array(data['centers'])
        aspect_ratios = np.array(data['aspect_ratios'])
        lengths = np.array(data['lengths'])
        
        # 绘制中心点密度
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.kdeplot(x=centers[:, 0], y=centers[:, 1], cmap="YlGnBu", fill=True, cbar=True, ax=ax)
        ax.set_title(f'{category.capitalize()} Center Point Density')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        plt.savefig(os.path.join(output_folder, f'{category}_center_density.png'))
        plt.close()
        
        # 绘制长宽比密度
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(aspect_ratios, kde=True, ax=ax)
        ax.set_title(f'{category.capitalize()} Aspect Ratio Density')
        ax.set_xlabel('Aspect Ratio')
        ax.set_ylabel('Density')
        plt.savefig(os.path.join(output_folder, f'{category}_aspect_ratio_density.png'))
        plt.close()
        
        # 绘制长度密度
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(lengths, kde=True, ax=ax)
        ax.set_title(f'{category.capitalize()} Length Density')
        ax.set_xlabel('Length')
        ax.set_ylabel('Density')
        plt.savefig(os.path.join(output_folder, f'{category}_length_density.png'))
        plt.close()
        
        # 绘制长宽比和长度的关系
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=aspect_ratios, y=lengths, ax=ax)
        model = aspect_ratio_length_models[category]
        x_range = np.linspace(aspect_ratios.min(), aspect_ratios.max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, color='red', linewidth=2)
        ax.set_title(f'{category.capitalize()} Aspect Ratio vs Length')
        ax.set_xlabel('Aspect Ratio')
        ax.set_ylabel('Length')
        plt.savefig(os.path.join(output_folder, f'{category}_aspect_ratio_length_relation.png'))
        plt.close()

def save_results(density_functions, aspect_ratio_length_models, coexistence_frequency, category_counts, output_folder):
    results = {
        'density_functions': density_functions,
        'aspect_ratio_length_models': aspect_ratio_length_models,
        'coexistence_frequency': coexistence_frequency,
        'category_counts': category_counts
    }
    
    with open(os.path.join(output_folder, 'analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

def calculate_coexistence_frequency(category_presence):
    coexistence_frequency = {}
    total_images = len(next(iter(category_presence.values())))
    
    for main_category in category_presence.keys():
        # 找出包含主类别的所有图像
        main_category_images = [i for i, present in enumerate(category_presence[main_category]) if present]
        main_category_count = len(main_category_images)
        
        if main_category_count > 0:
            # 计算在这些图像中其他类别的出现次数
            category_counts = {cat: sum(category_presence[cat][i] for i in main_category_images) 
                               for cat in category_presence.keys()}
            
            # 计算共存频率并归一化
            total_instances = sum(category_counts.values())
            coexistence_frequency[main_category] = {cat: count / total_instances 
                                                    for cat, count in category_counts.items()}
    
    return coexistence_frequency

def save_coexistence_frequency_to_excel(coexistence_frequency, output_folder):
    df = pd.DataFrame(coexistence_frequency).T
    df.index.name = '类别'
    
    excel_path = os.path.join(output_folder, '共存频率.xlsx')
    df.to_excel(excel_path)
    print(f"共存频率已保存到: {excel_path}")

def main():
    annotations_folder = 'annotations'
    output_folder = 'knowledge_graph'
    
    category_data, instance_counts, category_presence, category_counts = analyze_images(annotations_folder)
    density_functions, aspect_ratio_length_models = fit_density_functions(category_data, instance_counts)
    coexistence_frequency = calculate_coexistence_frequency(category_presence)
    
    plot_density_functions(category_data, density_functions, aspect_ratio_length_models, instance_counts, output_folder)
    save_results(density_functions, aspect_ratio_length_models, coexistence_frequency, category_counts, output_folder)
    
    save_coexistence_frequency_to_excel(coexistence_frequency, output_folder)

if __name__ == '__main__':
    main()