import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet_loss(nn.Module):
    def __init__(self, n_classes=7):
        super(ResNet_loss, self).__init__()

        self.n_classes = n_classes
        self.lambda_ = 0.3
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        if target[0][0] == 1 or target[0][1] == 1 or target[0][2] == 1:
            loss = self.lambda_ * self.BCEWithLogitsLoss(prediction, target)
        else:
            loss = self.BCEWithLogitsLoss(prediction, target)
        return loss