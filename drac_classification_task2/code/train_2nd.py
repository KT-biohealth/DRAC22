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
import control_model as control_model
import logging
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
import tools

import sys
#sys.path.append('/userdata/sungjinchoi/thyroid/db/input/pytorch-image-models-master')
#filterwarnings("ignore")\

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# TODO : set parameter to python files
import platform
if platform.system() == 'Linux':  # check if in Server
    print('yes!')
    DATA_PATH = r"/root/home/DRAC_2022/B_Image_Quality_Assessment/B. Image Quality Assessment/1. Original Images/input"
    os.environ['TORCH_HOME'] = '/root/home/DRAC_2022'
else:
    # DATA_PATH = r"C:\AI\DRAC\task3\input_5_fold"
    print('error in directory path')

IMAGE_PATH_TRAIN = 'train_images_pseudo'
CSV_NAME_TRAIN = 'train_images_pseudo.csv'
IMAGE_PATH_VAL = 'test_images'
CSV_NAME_VAL = 'test_8016.csv'
CSV_RESULT = '_train_result.csv'

STEP_SIZE = 3
seed = 2  # best seed : 2008

OUTPUT_FOLDER = 'ouput_NFNET_enhance_pseudo_SEED'+ str(seed)
CSV_IMAGE_ID = 'image_id'
CSV_LABEL_ID = 'label'

n_epochs = 30#100
batch_size = 4 # 64
init_lr = 1e-5

image_size = 448
num_workers = 0 #1
use_amp = True
PATIENCE = 10

def train_func(train_loader):
    model.train()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            model.set_feature(features)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

    scheduler.step()
    loss_train = np.mean(losses)
    return loss_train

def valid_func(valid_loader):
    model.eval()

    losses = []

    predict_list = []
    target_list = []


    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(valid_loader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            preds = torch.softmax(logits, dim=1)

            predict_list.extend(preds.cpu().tolist())
            target_list.extend(targets.detach().cpu().tolist())
            loss = criterion(logits, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])

    prediction = np.argmax(predict_list, axis=1).tolist()

    confusion_matrix = tools.Confusion_Matrix(target_list, prediction)
    cohen_kappa = tools.Cohen_Kappa(target_list, prediction)

    loss_valid = np.mean(losses)

    return loss_valid, cohen_kappa, confusion_matrix


if not torch.cuda.is_available():
    use_amp = False

model_dir = os.path.join(DATA_PATH, OUTPUT_FOLDER)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log = logging.getLogger()
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

file_handler = logging.FileHandler(f'{model_dir}/log.train.txt')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

start_time = time.strftime('%c', time.localtime(time.time()))
start_time_num = time.time()
log.info(f'training start time :{start_time}\n')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # for faster training, but not deterministic
    
seed_everything(seed)

y_true_result = []
y_pred_result = []

######################## Data load from cross validation folder ########################
print("Prepare Data Loader!!")

target_folder = DATA_PATH

print("Prepare Model!!")
# model = control_model.VisionTransformer_gigantic(out_dim=3)
model = control_model.NFNet_f6(out_dim=3)

model = model.to(device)

model_name = model.get_model_name()
image_size = model.get_input_shape()[-1]
image_channel = model.get_input_shape()[0]

criterion = nn.CrossEntropyLoss()
#criterion = tools.FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
#                                               lr_lambda=lambda epoch: 0.95 ** epoch,
#                                               last_epoch=-1,
#                                               verbose=False)
SUMMRAY = f'model type : {model_name}, batch size = {batch_size}, lr = {init_lr}, epochs = {n_epochs}, imag_size = {image_size}'
log.info(f'{SUMMRAY}\n')

transforms_train = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.Resize(image_size, image_size),
])

transforms_valid = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
])

image_train = os.path.join(target_folder, IMAGE_PATH_TRAIN)
image_val = os.path.join(target_folder, IMAGE_PATH_VAL)
df_train = pd.read_csv(os.path.join(target_folder, CSV_NAME_TRAIN))
df_val = pd.read_csv(os.path.join(target_folder, CSV_NAME_VAL))

dataset_train = control_data.ControlDataset(image_train,
                                            df_train[CSV_IMAGE_ID].values, df_train[CSV_LABEL_ID].values,
                                            transform=transforms_train)
dataset_val = control_data.ControlDataset(image_val,
                                            df_val[CSV_IMAGE_ID].values, df_val[CSV_LABEL_ID].values,
                                            transform=transforms_valid)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=True)

print("Start Training!!")
kappa_max_list = 0
loss_min_list = 9999999
loss_train_list = []
loss_val_list = []
kappa_list = []
roc_list = []
# early_stop = tools.Early_Stop()
early_stop = tools.EarlyStopping(patience=PATIENCE)
print("Early Stopping Ongoing, Patience is :", PATIENCE)

for epoch in range(0, n_epochs + 1):
    loss_train = train_func(train_loader)
    log.info(f'epoch: {epoch}/{n_epochs}, loss: {loss_train:.5f}')

    loss_valid, cohen_kappa, confusion_matrix = valid_func(val_loader)
    content_test = time.ctime() + ' ' + f'Test Results :loss_validation: {loss_valid:.5f}, kappa: {cohen_kappa:.4f}'
    log.info(f'{content_test}\n')
    if cohen_kappa >= kappa_max_list:
        log.info(f'save for best kappa\n')
        torch.save(model.state_dict(), f'{model_dir}/best_kappa.pth')
        kappa_max_list = cohen_kappa
    
    if loss_valid<= loss_min_list:
        log.info(f'save for best loss\n')
        torch.save(model.state_dict(), f'{model_dir}/best_loss.pth')
        loss_min_list = loss_valid

    if epoch == 0:
        result_data_frame = pd.DataFrame(
            {"train_loss": [loss_train], "val_loss": [loss_valid], "cohen_kappa": [cohen_kappa]})
    else:
        result_data_frame.loc[epoch] = [loss_train, loss_valid, cohen_kappa]

    early_stop.step(cohen_kappa)
    if early_stop.is_stop():
        print("early stopping is called")
        log.info(f'finish in {epoch}\n')
        break


log.info(f'save for last epoch\n')
torch.save(model.state_dict(), f'{model_dir}/latest.pth')
result_data_frame.to_csv(os.path.join(model_dir, CSV_RESULT), header=True, index=False)
print("finish")
