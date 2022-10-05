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

import sys
# sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
sys.path.append('/userdata/sungjinchoi/thyroid/db/input/pytorch-image-models-master')

import timm

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


class Densenet121(nn.Module):
    def __init__(self, model_name='densenet121', out_dim=3, pretrained=True, input_size=(3, 1024, 1024)):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=out_dim)

    def get_input_shape(self):
        return self.model.default_cfg['input_size']

    def get_model_name(self):
        return self.model_name

    def forward(self, x):
        output = self.model(x)

        return output


class BeitLarge(nn.Module):
    def __init__(self, model_name='beit_large_patch16_512', out_dim=3, pretrained=True):
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