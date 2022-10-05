import sys
import os
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import torch.nn.functional as F
import random
import cv2
import json


class ControlDataset(Dataset):
    def __init__(self, data_dir, image_data, label_data, transform=None, TARGET_VALUE=120):
        self.data_dir = data_dir
        self.image_data = image_data
        self.label_data = label_data
        self.transform = transform
        self.target_value = TARGET_VALUE

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        current_image_name = self.image_data[index]
        #current_image = cv2.imread(os.path.join(self.data_dir, current_image_name), cv2.IMREAD_COLOR)
        current_image = cv2.imread(os.path.join(self.data_dir, current_image_name), cv2.IMREAD_GRAYSCALE)
        current_image = cv2.bilateralFilter(current_image, -1, 60, 5)
        current_output = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
        # current_output = np.zeros((current_image.shape[0], current_image.shape[1], 3), dtype=np.uint8)
        # index_over_threshod = np.where(current_image >= self.target_value)
        # index_under_threshod = np.where(current_image < self.target_value)
        # current_output[:, :, 0] = current_image
        # current_output[index_over_threshod[0], index_over_threshod[1], 1] = current_image[index_over_threshod]
        # current_output[index_under_threshod[0], index_under_threshod[1], 2] = current_image[index_under_threshod]

        current_label = self.label_data[index]

        img = current_output / 255.0

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        label = torch.tensor(current_label).long()

        return torch.tensor(img).float(), label

        # 우선 기본으로 돌리도록 (k-fold만 적용해서)
        # 다음 부족한 label의 dataset을 늘려서 (vertical, horizontal flip, zoom-in (범위자동 지정 : mask를 보고), rotatio)

class TestDataset(Dataset):
    def __init__(self, data_dir, image_data, transform=None, TARGET_VALUE=120):
        self.data_dir = data_dir
        self.image_data = image_data
        self.transform = transform
        self.target_value = TARGET_VALUE

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        current_image_name = self.image_data[index]
        current_image = cv2.imread(os.path.join(self.data_dir, current_image_name), cv2.IMREAD_GRAYSCALE)
        current_image = cv2.bilateralFilter(current_image, -1, 60, 5)
        current_output = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)

        # current_output = np.zeros((current_image.shape[0], current_image.shape[1], 3), dtype=np.uint8)
        # index_over_threshod = np.where(current_image >= self.target_value)
        # index_under_threshod = np.where(current_image < self.target_value)
        # current_output[:, :, 0] = current_image
        # current_output[index_over_threshod[0], index_over_threshod[1], 1] = current_image[index_over_threshod]
        # current_output[index_under_threshod[0], index_under_threshod[1], 2] = current_image[index_under_threshod]

        img = current_output / 255.0
        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)

        return torch.tensor(img).float()

        # 우선 기본으로 돌리도록 (k-fold만 적용해서)
        # 다음 부족한 label의 dataset을 늘려서 (vertical, horizontal flip, zoom-in (범위자동 지정 : mask를 보고), rotatio)



if __name__ == '__main__':
    main()