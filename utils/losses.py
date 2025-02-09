import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) - logits (未归一化)
        # target: (B, H, W) - 类别索引
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        pred_softmax = F.softmax(pred, dim=1)

        intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + (1 - self.alpha) * self.dice(pred, target)


if __name__ == '__main__':
    criterion = DiceLoss()
    pred = torch.rand(1,3,4,4)
    target = torch.tensor(
            [[[1,2,0,0],
              [0,0,1,1],
              [0,2,0,0],
              [2,1,0,0]]]
    )
    dice = criterion(pred, target)
    print(dice)