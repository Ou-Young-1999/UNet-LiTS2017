import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
from model import UNet
from utils.losses import CombinedLoss
from utils.metrics import calculate_metrics
from PIL import Image
from tqdm import tqdm
import os
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, num_epochs, save_dir='./models'):
    best_dice = 0.0

    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        running_dice_1 = 0.0
        running_dice_2 = 0.0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                # 清零梯度
                optimizer.zero_grad()
                # 前向传播
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, labels)
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                # 更新进度条
                # 计算Dice分数和IoU
                pred = torch.argmax(outputs, dim=1)
                running_dices,_ = calculate_metrics(pred, labels)
                running_dice_1 += running_dices[0]
                running_dice_2 += running_dices[1]
                running_loss += loss.item()
                t.set_postfix(loss=running_loss / (t.n + 1), dice_1=running_dice_1 / (t.n + 1), dice_2=running_dice_2 / (t.n + 1))

        # 记录学习率
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        # 更新学习率
        scheduler.step()

        # 记录损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_dice_1 = running_dice_1 / len(train_loader)
        epoch_dice_2 = running_dice_2 / len(train_loader)

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_dice_1 = 0.0
        val_dice_2 = 0.0
        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = torch.argmax(outputs, dim=1)
                val_dices,_ = calculate_metrics(pred, labels)
                val_dice_1 += val_dices[0]
                val_dice_2 += val_dices[1]
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_dice_1 = val_dice_1 / len(val_loader)
        val_dice_2 = val_dice_2 / len(val_loader)

        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Dice1: {epoch_dice_1:.4f}, Train Dice2: {epoch_dice_2:.4f}')
        tqdm.write(f'Validation Loss: {val_loss:.4f}, Validation Dice1: {val_dice_1:.4f}, Validation Dice2: {val_dice_2:.4f}')

        writer.add_scalar('Loss/train', epoch_loss, epoch+1)
        writer.add_scalar('Dice_1/train', epoch_dice_1, epoch + 1)
        writer.add_scalar('Dice_2/train', epoch_dice_2, epoch + 1)

        writer.add_scalar('Loss/valid', val_loss, epoch+1)
        writer.add_scalar('Dice_1/valid', val_dice_1, epoch + 1)
        writer.add_scalar('Dice_2/valid', val_dice_2, epoch + 1)

        # 保存最优模型
        if val_dice_1+val_dice_2 > best_dice:
            best_dice = val_dice_1+val_dice_2
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            tqdm.write(f"Best model saved with Dice {best_dice:.4f}")
        time.sleep(0.5)

if __name__ == '__main__':
    # 设置设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to train...')

    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90, interpolation=Image.NEAREST),  # 标签使用最近邻插值
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    # 加载训练集和验证集
    train_dataset = CustomImageDataset(data_type='train', transform=transform)
    val_dataset = CustomImageDataset(data_type='valid', transform=valid_transform)
    print(f'Trainset size: {len(train_dataset)}')
    print(f'Validationset size: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    print(f'Trainloder size: {len(train_loader)}')
    print(f'Validationloader size: {len(val_loader)}')

    # 初始化SummaryWriter
    writer = SummaryWriter('runs/unet')
    writer.add_scalar('Loss/train', 10, 0)
    writer.add_scalar('Dice_1/train', 0, 0)
    writer.add_scalar('Dice_2/train', 0, 0)
    writer.add_scalar('Loss/valid', 10, 0)
    writer.add_scalar('Dice_1/valid', 0, 0)
    writer.add_scalar('Dice_2/valid', 0, 0)

    # 创建模型
    model = UNet(in_channels=3, num_classes=3)
    model = model.to(device)

    # 损失函数和优化器
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # StepLR：每 10 个 epoch 衰减学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 设置训练轮次
    num_epochs = 15

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, num_epochs)

    writer.close()
