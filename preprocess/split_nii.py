import os
import cv2
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 设置输入和输出文件夹
input_image_dir = 'F:\\LiTS\\volumes'  # 输入图像文件夹
input_label_dir = 'F:\\LiTS\\segmentations'  # 输入标签文件夹
output_dir = 'F:\\LiTS\\LiTS'  # 输出切片保存文件夹

# 调窗操作
def window_image(image, window_width, window_level):
    """
    调窗函数，适应指定的窗宽和窗位
    """
    min_intensity = window_level - window_width // 2
    max_intensity = window_level + window_width // 2

    # 限制像素值在 min_intensity 和 max_intensity 之间
    image = np.clip(image, min_intensity, max_intensity)

    # 将像素值归一化到 [0, 255]
    image = ((image - min_intensity) / (max_intensity - min_intensity) * 255).astype(np.uint8)

    return image

# 定义处理每个图像文件的函数
def process_image(image_file):
    print(f'Processing {image_file}')
    # 获取对应的标签文件路径
    label_file = image_file.replace('volume', 'segmentation')
    image_path = os.path.join(input_image_dir, image_file)
    label_path = os.path.join(input_label_dir, label_file)

    if not os.path.exists(label_path):
        print(f"标签文件 {label_file} 未找到，跳过该图像。")
        return

    # 读取图像和标签
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    image_data = image_nii.get_fdata()
    label_data = label_nii.get_fdata()

    # 创建对应图像文件夹
    image_output_dir = os.path.join(output_dir, image_file.split('.')[0].split('-')[1], 'Image')
    label_output_dir = os.path.join(output_dir, image_file.split('.')[0].split('-')[1], 'GT')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # 切片并保存
    for i in range(image_data.shape[2]):  # 假设按 z 轴切片
        image_slice = image_data[:, :, i]
        label_slice = label_data[:, :, i]

        # 检查标签切片中是否有非零值
        if np.any(label_slice != 0):  # 如果标签切片有非零值
            # 保存切片图像
            image_slice_path = os.path.join(image_output_dir, f'{i}.png')
            label_slice_path = os.path.join(label_output_dir, f'{i}.png')

            # 将 1 转换为 128
            label_slice[label_slice == 1] = 128
            # 将 2 转换为 255
            label_slice[label_slice == 2] = 255
            # 归一化标签
            normalized_label_slice = label_slice.astype(np.uint8)

            # 归一化图像
            normalized_image_slice = window_image(image_slice, window_width=400, window_level=50)

            # 保存图像和标签切片
            cv2.imwrite(image_slice_path, normalized_image_slice)
            cv2.imwrite(label_slice_path, normalized_label_slice)

if __name__ == '__main__':

    # 获取所有图像文件路径
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # 创建一个线程池，定义线程数量
    num_threads = 1  # 线程数，可以根据你的 CPU 核心数来调整
    executor = ThreadPoolExecutor(max_workers=num_threads)

    # 使用多线程批量处理图像文件
    for image_file in image_files:
        executor.submit(process_image, image_file)

    # 关闭线程池
    executor.shutdown(wait=True)

    print("所有图像的切片保存完毕！")
