import os
import numpy as np
import pandas as pd
import shutil
import cv2 as cv
import custom_augmentation
import platform
if platform.system() =='Linux': #check if in Server
    RAW_FOLDER = r"/userdata/common/MICCAI2022/DRAC/task3/input_new_5_fold/fold_0"
else:
    RAW_FOLDER = r"C:\AI\DRAC\task3\input_5_fold\fold_0"
INPUT_FOLDER = os.path.join(RAW_FOLDER, "train_images")
INPUT_CSV = os.path.join(RAW_FOLDER, "train.csv")
OUTPUT_FOLDER = os.path.join(RAW_FOLDER, "train_images_balance_aug_random")
OUTPUT_CSV = os.path.join(RAW_FOLDER, "train_balance_aug_random.csv")

STR_IMAGE_NAME_IN_CSV = 'image_id'
STR_LABEL_IN_CSV = 'label'

def calculate_data_difference(label_input):
    # find maximum count and remain numbers
    label_count = label_input.value_counts()
    count_list = []
    maximum_count = 0
    for index in range(0, len(label_count)):
        current_count = label_count[index]
        count_list.append(current_count)
        maximum_count = max(maximum_count, current_count)

    difference_list = []
    for current_count in count_list:
        difference_list.append(abs(maximum_count - current_count))

    return count_list, difference_list

def make_balance_dataset(input_image, input_label):
    # repeat image data and make label (copy
    label_count, label_diff = calculate_data_difference(input_label)

    maximum_count = np.max(label_count)
    balance_count = maximum_count * len(label_count)
    count_label = len(label_count)
    images = input_image.values

    result_image_name = []
    for current_label in range(0, count_label):
        #split image_id depending on label
        current_index_labels = np.where(input_label.values == current_label)
        current_image_labels = images[current_index_labels]

        if label_diff[current_label] > 0:
            current_image_labels = np.pad(current_image_labels, (0, abs(maximum_count - label_count[current_label])), 'wrap')
        result_image_name.append(current_image_labels)

    return result_image_name, np.sum(label_diff)

def make_augmentation_random(data_size, MAXIMUM_ANGLE=5, MAXIMUM_ZOOM_IN=1.2, MAXIMUM_BILATERAL_INDEX=10):
    output_list = []
    value_v_flip = custom_augmentation.get_uniform_random(low=0, high=1, size=data_size)
    value_h_flip = custom_augmentation.get_uniform_random(low=0, high=1, size=data_size)
    value_rotation = custom_augmentation.get_range_random(low=-MAXIMUM_ANGLE, high=MAXIMUM_ANGLE, size=data_size)  # center based
    value_zoom_in = custom_augmentation.get_half_random(low=1, high=MAXIMUM_ZOOM_IN, size=data_size)  # zoom-in (ROI center)
    value_sharpen = custom_augmentation.get_half_random(low=0, high=1, size=data_size)
    value_bilateral = custom_augmentation.get_half_random(low=0, high=MAXIMUM_BILATERAL_INDEX, size=data_size)  # image filter
    value_gamma = custom_augmentation.get_range_random(low=0.5, high=2, size=data_size)  # for contrast and brightness (0.5 ~ 2로...) 1 ~ 2로하고, 1/1 ~ 1/2까지..(-1에서 1로하고... 변환에서 쓸까)

    output_list.append(value_v_flip)
    output_list.append(value_h_flip)
    output_list.append(value_rotation)
    output_list.append(value_zoom_in)
    output_list.append(value_sharpen)
    output_list.append(value_bilateral)
    output_list.append(value_gamma)

    return output_list
def do_augmentation(input_name, output_name, data_number, augmentation_values):
    input_image = cv.imread(input_name)

    # Geometric Augmentation
    image_v_flip = custom_augmentation.flip_vertical(input_image, augmentation_values[0][data_number])
    image_h_flip = custom_augmentation.flip_horizontal(image_v_flip, augmentation_values[1][data_number])
    image_rotation = custom_augmentation.rotate_image_on_center(image_h_flip, augmentation_values[2][data_number])
    image_zoom_in = custom_augmentation.zoom_in_center(image_rotation, augmentation_values[3][data_number])
    image_sharpen = custom_augmentation.shapren_filter(image_zoom_in, augmentation_values[4][data_number])
    image_bilateral = custom_augmentation.bilateral_filter(image_sharpen, augmentation_values[5][data_number])
    output_image = custom_augmentation.gamma_correction(image_bilateral, augmentation_values[6][data_number])
    cv.imwrite(output_name, output_image)

def copy_and_append_augmentation(src_folder, dest_folder, image_names_list, balance_count):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    is_made_result = False
    augmentation_values = make_augmentation_random(balance_count)
    count_label = len(image_names_list)
    total_index = 0
    adding_index = 0
    for current_label in range(0, count_label):
        current_image_names = image_names_list[current_label]
        for index, current_image_name in enumerate(current_image_names):
            current_input_name = os.path.join(src_folder, current_image_name)
            packet = current_image_name.split(".")[0]
            ext = current_image_name.split(".")[-1]

            original_packet = packet
            existing_number = 0
            while os.path.exists(os.path.join(dest_folder, packet + "." + ext)):
                existing_number += 1
                packet = original_packet + "_"+ str(existing_number)

            current_output = packet + "." + ext
            current_output_name = os.path.join(dest_folder, current_output)
            if existing_number <= 0:
                shutil.copy(current_input_name, current_output_name)
            else:
                do_augmentation(current_input_name, current_output_name, adding_index, augmentation_values)
                adding_index += 1
            total_index += 1
            print(current_output)
            if not is_made_result:
                result_data_frame = pd.DataFrame({STR_IMAGE_NAME_IN_CSV:[current_output], STR_LABEL_IN_CSV:[current_label]})
                is_made_result = True
            else:
                result_data_frame.loc[total_index] = [current_output, current_label]

    return result_data_frame

data_frame = pd.read_csv(INPUT_CSV)
input_image = data_frame[STR_IMAGE_NAME_IN_CSV]
input_label = data_frame[STR_LABEL_IN_CSV]

balanced_image_name, balance_count = make_balance_dataset(input_image, input_label)
balance_data_frame = copy_and_append_augmentation(INPUT_FOLDER, OUTPUT_FOLDER, balanced_image_name, balance_count)
balance_data_frame.to_csv(OUTPUT_CSV, header=True, index=False)
#shutil.copytree(INPUT_FOLDER, OUTPUT_FOLDER)

print("total data number = " + str(len(input_label)))
