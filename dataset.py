from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

label_mapping = {
    0: 0,     # 背景 -> 类别0
    128: 1,   # 器官A -> 类别1
    255: 2    # 器官B -> 类别2
}
# 逆向映射（用于预测结果）
inverse_mapping = {v: k for k, v in label_mapping.items()}

# 定义自己的数据集类
class CustomImageDataset(Dataset):
    def __init__(self, data_type, transform=None):
        self.img_dir = 'F:\\LiTS\\LiTS'
        self.data_type = data_type
        self.transform = transform
        self.image_paths = []
        self.label_paths = []

        # 将文件路径和标签存储在列表中
        txt_path = './preprocess/' + self.data_type + '.txt'
        with open(txt_path, 'r') as txt_file:
            for i, row in enumerate(txt_file):
                image_case = os.path.join(self.img_dir, row.split('\n')[0], 'Image')
                label_case = os.path.join(self.img_dir, row.split('\n')[0], 'GT')
                for case in os.listdir(image_case):
                    self.image_paths.append(os.path.join(image_case, case))
                    self.label_paths.append(os.path.join(label_case, case))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 确保图像是RGB格式
        label = Image.open(label_path).convert("L")

        if self.transform:
            # 固定全局随机种子和可变索引以保证可复现性和随机性
            seed = 3407+idx

            # 对图像应用变换
            torch.manual_seed(seed)
            image = self.transform(image)

            # 对标签应用相同变换（禁用插值，使用最近邻，防止标签变化）
            torch.manual_seed(seed)
            label = self.transform(label)

        # 转为Tensor
        image = transforms.ToTensor()(image)  # (C, H, W)

        # 转换为Numpy数组
        label = np.array(label)
        # 映射标签值到连续索引
        label = np.vectorize(label_mapping.get)(label)
        label = torch.from_numpy(label).long()  # (H, W)

        return image, label

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化

if __name__ == '__main__':
    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}')

    # 定义图像变换
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90, interpolation=Image.NEAREST),  # 标签使用最近邻插值
        transforms.Resize((256,256), interpolation=Image.NEAREST),
    ])

    # 创建数据集实例
    train_dataset = CustomImageDataset(data_type='train', transform=transform)
    validation_dataset = CustomImageDataset(data_type='valid', transform=transform)
    print(f'train size: {len(train_dataset)}')
    print(f'valid size: {len(validation_dataset)}')

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    # 获取一个批次的数据
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    images = (images * 255).to(torch.uint8) # [0,1] to [0,255]
    labels = torch.from_numpy(np.vectorize(inverse_mapping.get)(labels.numpy())) # inverse mapping
    labels = torch.stack((labels, labels, labels), dim=1) # Gray to RGB

    grid_img = make_grid(images)
    grid_label = make_grid(labels)
    concat = torch.cat((grid_img, grid_label), dim=1)

    # 可视化网格图片
    plt.imshow(concat.permute(1, 2, 0))  # 调整通道顺序以适应 matplotlib 的要求
    plt.show()