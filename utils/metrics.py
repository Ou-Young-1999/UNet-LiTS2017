import torch


# 计算指标函数
def calculate_metrics(preds, targets, n_classes=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dice_scores = []
    iou_scores = []

    for cls in range(1, n_classes):  # 忽略背景
        num = targets.shape[0]
        dice_sum = torch.tensor([0.0]).to(device)
        iou_sum = torch.tensor([0.0]).to(device)
        for i in range(0, num):  # 对batch中的每个实例进行计算
            pred = preds[i]
            target = targets[i]
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            intersection = torch.logical_and(pred_cls, target_cls).sum().float()
            union = torch.logical_or(pred_cls, target_cls).sum().float()

            if intersection + union > 0:
                dice = (2. * intersection) / (pred_cls.sum() + target_cls.sum() + 1e-6)
                iou = intersection / (union + 1e-6)
            else:
                dice = torch.tensor([1.0]).to(device)
                iou = torch.tensor([1.0]).to(device)
            dice_sum += dice
            iou_sum += iou
        dice_per = dice_sum / num
        iou_per = iou_sum / num

        dice_scores.append(dice_per.item())
        iou_scores.append(iou_per.item())

    return dice_scores, iou_scores

if __name__ == '__main__':
    pred = torch.rand(2,3,4,4)
    pred = torch.argmax(pred, dim=1)
    target = torch.tensor(
            [[[1, 2, 0, 0],
              [0, 0, 1, 1],
              [0, 2, 0, 0],
              [2, 1, 0, 0]],
             [[1, 2, 0, 0],
              [0, 0, 1, 1],
              [0, 2, 0, 0],
              [2, 1, 0, 0]]
             ]
    )
    dice, iou = calculate_metrics(pred, target)
    print(dice)
    print(iou)