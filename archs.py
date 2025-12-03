import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from torchvision.models import vit_b_32, ViT_B_32_Weights

class WRN_28_10(WideResNet):
    def __init__(self, num_classes):
        super().__init__(depth=28, widen_factor=10, num_classes=num_classes)

class WRN_40_2(WideResNet):
    def __init__(self, num_classes):
        super().__init__(depth=40, widen_factor=2, num_classes=num_classes)

class ViT_B32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = vit_b_32(weights = ViT_B_32_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Linear(768, num_classes, bias=True)
        self.model.num_classes = num_classes
