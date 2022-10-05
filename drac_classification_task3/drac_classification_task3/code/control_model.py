import os
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
#import timm
import cv2

import timm

class Resnet50D(nn.Module):
    def __init__(self, model_name='resnet50d', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class SE(nn.Module):
    
    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class Resnet50(nn.Module):
    def __init__(self, model_name='resnet50', out_dim=1, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output


class Resnet101D(nn.Module):
    def __init__(self, model_name='resnet101d', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class BeitLarge(nn.Module):
    def __init__(self, model_name='beit_large_patch16_512', out_dim=1, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim) # drop 시 drop_rate=0.3

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class VoloLarge(nn.Module):
    def __init__(self, model_name='volo_d5_512', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class VisionTransformer_large_384(nn.Module):
    def __init__(self, model_name='vit_large_patch16_384', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class VisionTransformer_gigantic(nn.Module):
    def __init__(self, model_name='vit_gigantic_patch14_224', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output
    
class VisionTransformer_giant(nn.Module):
    def __init__(self, model_name='vit_giant_patch14_224', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class HybridVisionTransformerV2(nn.Module):
    def __init__(self, model_name='vit_base_r50_s16_384', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class HybridVisionTransformerV2_large(nn.Module):
    def __init__(self, model_name='vit_large_r50_s32_384', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class EffnetB7(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b7', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class EffnetB7_NS(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b7_ns', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class EffnetL2_NS(nn.Module):
    def __init__(self, model_name='tf_efficientnet_l2_ns', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class NFNet_f6(nn.Module):
    def __init__(self, model_name='dm_nfnet_f6', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output

class NFNet_f7(nn.Module):
    def __init__(self, model_name='nfnet_f7', out_dim=3, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output


if __name__ == '__main__':
    main()