import csv
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import CustomImageDataset
from model import UNet
from tqdm import tqdm
import os
import random
import numpy as np
from PIL import Image

from utils.losses import CombinedLoss
from utils.metrics import calculate_metrics

label_mapping = {
    0: 0,     # 背景 -> 类别0
    128: 1,   # 器官A -> 类别1
    255: 2    # 器官B -> 类别2
}
# 逆向映射（用于预测结果）
inverse_mapping = {v: k for k, v in label_mapping.items()}

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化


def save_image(images, preds, labels, name, save_dir='./results/img_pred_gt'):
    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = images.to('cpu')
    preds = preds.to('cpu')
    labels = labels.to('cpu')

    images = (images * 255).to(torch.uint8)  # [0,1] to [0,255]
    preds = torch.from_numpy(np.vectorize(inverse_mapping.get)(preds.numpy()))  # inverse mapping
    preds = torch.stack((preds, preds, preds), dim=1).to(torch.uint8)   # Gray to RGB
    labels = torch.from_numpy(np.vectorize(inverse_mapping.get)(labels.numpy()))  # inverse mapping
    labels = torch.stack((labels, labels, labels), dim=1).to(torch.uint8)   # Gray to RGB

    grid_img = make_grid(images)
    grid_pred = make_grid(preds)
    grid_label = make_grid(labels)
    concat = torch.cat((grid_img, grid_pred, grid_label), dim=1)

    # PyTorch张量的维度是[C, H, W]，而PIL Image需要的是[H, W, C]
    # 因此，需要将张量维度从[C, H, W]转换为[W, H, C]
    concat = concat.permute(2, 1, 0)

    # 将张量转换为PIL图像
    concat = Image.fromarray(concat.numpy())

    # 保存图像
    concat.save(os.path.join(save_dir,name))

# 训练函数
def test_model(model, test_loader, criterion, save_dir='./results'):
    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()  # 设置模型为评估模式
    test_dice_1 = []
    test_dice_2 = []
    test_iou_1 = []
    test_iou_2 = []
    test_loss = 0.0
    i = 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)
            test_dices, test_ious = calculate_metrics(preds, labels)
            test_dice_1.append(test_dices[0])
            test_dice_2.append(test_dices[1])
            test_iou_1.append(test_ious[0])
            test_iou_2.append(test_ious[1])

            if i < 1000:
                name = str(i)+'.png'
                save_image(inputs, preds, labels, name)
                i += 1

            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    dice1 = sum(test_dice_1) / len(test_dice_1)
    dice2 = sum(test_dice_2) / len(test_dice_2)
    iou1 = sum(test_iou_1) / len(test_iou_1)
    iou2 = sum(test_iou_2) / len(test_iou_2)

    tqdm.write(f'Test Loss: {test_loss:.4f}, Dice1: {dice1:.4f}, Dice2: {dice2:.4f}, IoU1: {iou1:.4f}, IoU2: {iou2:.4f}')

    # 保存为 CSV 文件
    with open(os.path.join(save_dir, 'score.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入列标题（可选）
        writer.writerow(['Dice1', 'Dice2', 'IoU1', 'IoU2'])
        # 将列表转换为列，并写入CSV文件
        for item1, item2, item3, item4 in zip(test_dice_1, test_dice_2, test_iou_1, test_iou_2):
            writer.writerow([item1, item2, item3, item4])
    print(f"Dice score and IoU score have saved to {os.path.join(save_dir, 'score.csv')}")

    return test_dice_1, test_dice_2, test_iou_1, test_iou_2

if __name__ == '__main__':
    # 设置设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to test...')

    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}')

    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    # 加载测试集
    test_dataset = CustomImageDataset(data_type='test', transform=test_transform)
    print(f'Testset size: {len(test_dataset)}')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 导入模型
    print(f'Loading model...')
    model = UNet(in_channels=3, num_classes=3)
    model = model.to(device)
    model.load_state_dict(torch.load('./models/best.pth'))

    # 损失函数和优化器
    criterion = CombinedLoss()

    # 训练模型
    test_dice_1, test_dice_2, test_iou_1, test_iou_2 = test_model(model, test_loader, criterion)