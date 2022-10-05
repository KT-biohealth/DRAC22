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
import control_data_coloring as control_data
import control_model
import logging
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
import tools
from tools_criterion import get_criterion
from tools_scheduler import get_scheduler, step_scheduler
from tqdm import tqdm

import math
from torch.optim.lr_scheduler import _LRScheduler
import sys


'''
[------------------------------best parameter------------------------------]
model = Beit Large 
image size = 512
batch size = 4 
SEED = 1234
epoch = 30 
DATASET = Add Random augmentation (path = ../db/input_new_5_fold/fold_0/train_images_balance_aug_random)
FOLD = 5
criterion = CrossEntropyLoss
optimizer = AdamW
scheduler = StepLR(step size=3, lr=1e-5, gamma=0.5)
use amp = true
[--------------------------------------------------------------------------]
'''

# select_scheduler='GradualWarmupSchedulerV2' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'GradualWarmupSchedulerV2']
select_criterion='CrossEntropyLoss' # ['CrossEntropyLoss', LabelSmoothing', 'FocalLoss' 'FocalCosineLoss', 'SymmetricCrossEntropyLoss', 'BiTemperedLoss', 'TaylorCrossEntropyLoss']


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# TODO : set parameter to python files
import platform
if platform.system() =='Linux': #check if in Server
    DATA_PATH = r"/home/miccai_drac_classification/db/input_5_fold" 
    os.environ['TORCH_HOME'] = '/home/miccai_drac_classification/db/timm_model'
else:
    DATA_PATH = r"C:\AI\DRAC\task3\input_5_fold"   # pc local path

IMAGE_PATH_TRAIN = 'train_images_balance_aug_random'     # train image path
CSV_NAME_TRAIN = 'train_balance_aug_random.csv'          # train image csv
IMAGE_PATH_VAL = 'val_images'                            # valid image path
CSV_NAME_VAL = 'val.csv'                                 # valid image csv
CSV_RESULT = '_train_result.csv'                         # train result

OUTPUT_FOLDER = 'output'                                 # output folder
CSV_IMAGE_ID = 'image_id'                                # train csv columns 1
CSV_LABEL_ID = 'label'                                   # train csv columns 2

#TAGET_FOLD = 2                                          # FOLD 지정
STEP_SIZE = 3
NFOLD = 5
START_FOLD = 0
END_FOLD = 4
# END_FOLD = min(END_FOLD, NFOLD-1)

n_epochs = 30
batch_size = 4 
init_lr = 1e-5

seed = 1234 
num_workers = 0 #1
use_amp = True
early_stop = tools.Early_Stop()

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
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())

    scheduler.step()
    loss_train = np.mean(losses)
    return loss_train

def valid_func(valid_loader):
    model.eval()
    losses = []
    predict_list = []
    target_list = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            preds = torch.softmax(logits, dim=1)

            predict_list.extend(preds.cpu().tolist())
            target_list.extend(targets.detach().cpu().tolist())
            loss = criterion(logits, targets)
            losses.append(loss.item())

    prediction = np.argmax(predict_list, axis=1).tolist()

    confusion_matrix = tools.Confusion_Matrix(target_list, prediction)
    cohen_kappa = tools.Cohen_Kappa(target_list, prediction)
    roc_auc = tools.ROC_AUC(target_list, predict_list)
    loss_valid = np.mean(losses)

    return loss_valid, roc_auc, cohen_kappa, confusion_matrix


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

file_handler = logging.FileHandler(f'{model_dir}/log.train_{START_FOLD}_{END_FOLD}.txt')
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

for fold_num in range(START_FOLD, END_FOLD + 1):
    target_folder = os.path.join(DATA_PATH, "fold_" + str(fold_num))

    print("Prepare Model!!")
    model = control_model.BeitLarge(out_dim=3)
    model = model.to(device)

    model_name = model.get_model_name()
    image_size = model.get_input_shape()[-1]
    image_channel = model.get_input_shape()[0]

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    

    criterion = get_criterion(select_criterion, 3, device)
    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    # scheduler = get_scheduler(select_scheduler, optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5)


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
    print("train_loader :", len(train_loader))
    print("val_loader :", len(val_loader))

    kappa_max_list = 0
    val_min_los=999999
    loss_train_list = []
    loss_val_list = []
    kappa_list = []
    roc_list = []

    early_stop_counter=0
    best_valid_loss=np.inf
    patience=5

    for epoch in range(0, n_epochs + 1): #range(0, n_epochs + 1)
        loss_train = train_func(train_loader)
        log.info(f'fold: {fold_num}, epoch: {epoch}/{n_epochs}, loss: {loss_train:.5f}') 

        loss_valid, roc_auc, cohen_kappa, confusion_matrix = valid_func(val_loader) #roc_auc
        print(f'======{epoch}th Confusion matrix=============')
        print(confusion_matrix)
        lr_value=optimizer.param_groups[0]['lr']
        content_test = time.ctime() + ' ' + f'Test Results :loss_validation: {loss_valid:.5f}, kappa: {cohen_kappa:.4f}, lr: {lr_value:.10f}'
        log.info(f'{content_test}\n') 

        if cohen_kappa >= kappa_max_list:
            log.info(f'save for best kappa\n')
            torch.save(model.state_dict(), f'{model_dir}/{fold_num}_fold_best_kappa.pth')
            kappa_max_list = cohen_kappa

        if loss_valid <= val_min_los:
            log.info(f'save for minimum validation loss\n')
            torch.save(model.state_dict(), f'{model_dir}/{fold_num}_fold_minimum_loss.pth')
            val_min_los = loss_valid
            early_stop_counter=0

        if epoch == 0:
            result_data_frame = pd.DataFrame(
                {"train_loss": [loss_train], "val_loss": [loss_valid], "cohen_kappa": [cohen_kappa],  "AUC_ROC":[roc_auc], "Learning_rate":[lr_value], "Confusion_mat":[confusion_matrix]})
        else:
            result_data_frame.loc[epoch] = [loss_train, loss_valid, cohen_kappa, roc_auc, lr_value, confusion_matrix]


        # if early_stop.is_stop_early(cohen_kappa):
        #     log.info(f'finish in {epoch}\n')        
        #     break


    log.info(f'save for last epoch\n')
    #torch.save(model.state_dict(), f'{model_dir}/{fold_num}_fold_latest.pth')
    result_data_frame.to_csv(os.path.join(model_dir, str(fold_num) + CSV_RESULT), header=True, index=False)

print("finish")