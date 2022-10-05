import os
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold


RAW_INPUT_FOLDER = r"/home/drac_classification_task3/db/input"
INPUT_FOLDER = os.path.join(RAW_INPUT_FOLDER, "train_images")
INPUT_CSV = os.path.join(RAW_INPUT_FOLDER, "train.csv")

FOLD_NUMBER = 5
RAW_OUTPUT_FOLDER = r"/home/drac_classification_task3/db/input_" + str(FOLD_NUMBER) + "_fold"

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def copy_file_by_names(file_names, src_folder, dest_folder):
    check_folder(dest_folder)

    for current_file in file_names:
        shutil.copy(os.path.join(src_folder, current_file), os.path.join(dest_folder, current_file))

def separate_save_dataframe(data_frame, index, csv_name, target_folder):
    check_folder(target_folder)

    target_csv_name = os.path.join(target_folder, csv_name)
    target_data_frame = data_frame.iloc[index]
    target_data_frame.to_csv(target_csv_name, header=True, index=False)

data_frame = pd.read_csv(INPUT_CSV)
data_frame.columns = ['image_id', 'label']

input_image = data_frame['image_id']
input_label = data_frame['label']
k_fold = StratifiedKFold(n_splits=FOLD_NUMBER, shuffle=True)
for fold,(train_idx,val_idx) in enumerate(k_fold.split(input_image, input_label)):
    print("current fold = " + str(fold))
    current_output_folder = os.path.join(RAW_OUTPUT_FOLDER, "fold_" + str(fold))

    current_train_folder = os.path.join(current_output_folder, "train_images")
    copy_file_by_names(input_image.values[train_idx], INPUT_FOLDER, current_train_folder)
    current_val_folder = os.path.join(current_output_folder, "val_images")
    copy_file_by_names(input_image.values[val_idx], INPUT_FOLDER, current_val_folder)

    separate_save_dataframe(data_frame, train_idx, "train.csv", current_output_folder)
    separate_save_dataframe(data_frame, val_idx, "val.csv", current_output_folder)



