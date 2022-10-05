import os
import pandas as pd
import numpy as np
import time
import random
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import albumentations
from sklearn.metrics import roc_auc_score
from sklearn import metrics, neighbors
import control_data_enhance as control_data
import control_model
import logging
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
import tools

import sys
#sys.path.append('/userdata/sungjinchoi/thyroid/db/input/pytorch-image-models-master')
#filterwarnings("ignore")\

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO : set parameter to python files
import platform
if platform.system() =='Linux': #check if in Server
    DATA_PATH = r"/root/home/DRAC_2022/B_Image_Quality_Assessment/B. Image Quality Assessment/1. Original Images/input"
    MODEL_PATH = "/root/home/DRAC_2022/B_Image_Quality_Assessment/B. Image Quality Assessment/1. Original Images/input/ouput_NFNET_enhance_pseudo_SEED270001"
else:
    # DATA_PATH = r"C:\AI\DRAC\task3\input_5_fold"
    print("error is ocurred")

IMAGE_PATH_VAL = 'test_images'
CSV_INPUT = 'submission_result.csv'
CSV_RESULT = 'v10.1.csv'

CSV_IMAGE_ID = 'case'
CSV_LABEL_ID = 'class'
CSV_PO_ID = 'P0'
CSV_P1_ID = 'P1'
CSV_P2_ID = 'P2'

fold_id = 0
image_size = 224
seed = 1234 # best seed : 2008
warmup_epo = 1
init_lr = 1e-5
batch_size = 8 # 64
valid_batch_size = 32
n_epochs = 100#100
warmup_factor = 10
num_workers = 0 #1
use_amp = True
debug = True # change this to run on full data
early_stop = 20
val_save_time = int(n_epochs / 10)
NFOLD = 5

def valid_func(test_loader):
    model.eval()

    predict_list = []
    datalen = test_loader.__len__()
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            print(str(batch_idx) + "/" + str(datalen))
            images = images.to(device)
            logits = model(images)
            preds = torch.softmax(logits, dim=1)

            predict_list.extend(preds.cpu().tolist())

    prediction = np.argmax(predict_list, axis=1).tolist()

    return prediction, predict_list

if not torch.cuda.is_available():
    use_amp = False

model_dir = os.path.join(MODEL_PATH)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log = logging.getLogger()
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

file_handler = logging.FileHandler(f'{model_dir}/log.validation.txt')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

start_time = time.strftime('%c', time.localtime(time.time()))
start_time_num = time.time()
log.info(f'validation start time :{start_time}\n')

y_true_result = []
y_pred_result = []

######################## Data load from cross validation folder #######################
print("Prepare Data Loader!!")
softmax_result_list = []
model = control_model.NFNet_f6(out_dim=3, pretrained=False)

print("For Best Kappa")
print("Load Model!!")
    
model.load_state_dict(torch.load(f'{model_dir}/best_kappa.pth'))
model.eval()
model = model.to(device)

model_name = model.get_model_name()
image_size = model.get_input_shape()[-1]
image_channel = model.get_input_shape()[0]

transforms_valid = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
])
target_folder = os.path.join(DATA_PATH)
image_val = os.path.join(target_folder, IMAGE_PATH_VAL)
df_val = pd.read_csv(os.path.join(target_folder, CSV_INPUT))
dataset_val = control_data.TestDataset(image_val, df_val[CSV_IMAGE_ID].values,
                                        transform=transforms_valid)

val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=True)

print("Start validation!!")
prediction_list, softmax_result = valid_func(val_loader)
softmax_result_list.append(softmax_result)

print("For Best loss")
print("Load Model!!")
    
model.load_state_dict(torch.load(f'{model_dir}/best_loss.pth'))
model.eval()
model = model.to(device)
print("Start validation!!")
prediction_list, softmax_result = valid_func(val_loader)
softmax_result_list.append(softmax_result)

result_np = np.array(softmax_result_list)
result_avg = np.mean(result_np, axis=0)
prediction = np.argmax(result_avg, axis=1)
P0 = result_avg[:, 0]
P1 = result_avg[:, 1]
P2 = result_avg[:, 2]
name = df_val[CSV_IMAGE_ID].values
result_data_frame = pd.DataFrame({CSV_IMAGE_ID: name, CSV_LABEL_ID: prediction, CSV_PO_ID: P0, CSV_P1_ID: P1, CSV_P2_ID: P2})
result_data_frame.to_csv(os.path.join(MODEL_PATH, CSV_RESULT), header=True, index=False)
print("Finish")