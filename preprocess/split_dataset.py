import os
import random
import cv2
import numpy as np

def split_data(files):
    # 确保随机性
    random.seed(3407)
    random.shuffle(files)

    # 计算索引位置
    total_size = len(files)
    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)

    # 划分数据集
    train_set = files[:train_size]
    val_set = files[train_size:train_size + val_size]
    test_set= files[train_size + val_size:]

    return train_set, val_set, test_set

def statistical_dataset(data):
    dataset_size = 0
    root_path = 'F:\\LiTS\\LiTS'
    pixel_counts = {0: 0, 128: 0, 255: 0}
    for case in data:
        case_path = os.path.join(root_path, case, 'GT')
        sub_path = os.listdir(case_path)
        dataset_size += len(sub_path)
        for sub in sub_path:
            img_path = os.path.join(case_path, sub)
            # 读取图片
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # 计算每个像素值的数量
            # count_0 = np.sum(image == 0)
            count_128 = np.sum(image == 128)
            count_255 = np.sum(image == 255)
            # pixel_counts[0] += count_0
            pixel_counts[128] += count_128
            pixel_counts[255] += count_255
    return dataset_size, pixel_counts

def write_txt(data, txt_path):
    # 写入到txt
    with open(txt_path, 'w') as file:
        for case in data[:len(data)-1]:
            file.write(case + '\n')
        file.write(data[-1]) # 最后一行不需要换行
    print(f'writing {txt_path}')

if __name__ == '__main__':
    root_path = 'F:\\LiTS\\LiTS'
    files = os.listdir(root_path)
    print(f'total size: {len(files)}')
    train_set, val_set, test_set = split_data(files)
    print(f'train size: {len(train_set)}')
    print(f'valid size: {len(val_set)}')
    print(f'test size: {len(test_set)}')
    train_size, train_label = statistical_dataset(train_set)
    print(f'train image size: {train_size}, train label statistical: {train_label}')
    valid_size, valid_label = statistical_dataset(val_set)
    print(f'valid image size: {valid_size}, valid label statistical: {valid_label}')
    test_size, test_label = statistical_dataset(test_set)
    print(f'test image size: {test_size}, test label statistical: {test_label}')
    write_txt(train_set, 'train.txt')
    write_txt(val_set, 'valid.txt')
    write_txt(test_set, 'test.txt')