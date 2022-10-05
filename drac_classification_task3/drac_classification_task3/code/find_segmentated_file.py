import numpy as np
import os
import cv2 as cv
import pandas as pd

RAW_PATH = r"C:\AI\DRAC\task3\post_processing\Task3_testimage_result_0907\MICCAI_sum"
TARGET_CLASS = 2
INPUT_PATH = os.path.join(RAW_PATH, str(TARGET_CLASS))
OUTPUT_PATH = os.path.join(RAW_PATH, str(TARGET_CLASS) + "_mask")
OUTPUT_AREA_CSV = os.path.join(RAW_PATH, str(TARGET_CLASS) + "_area.csv")

THRESHOLD_1 = 100
THRESHOLD_2 = 0
THRESHOLD_3 = 100

CSV_ID_IMAGE = 'image_id'
CSV_ID_AREA = 'area'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

file_list = os.listdir(INPUT_PATH)
image_list = []
area_list = []

for current_file in file_list:
    current_input_file = os.path.join(INPUT_PATH, current_file)
    current_input = cv.imread(current_input_file, cv.IMREAD_GRAYSCALE)

    if TARGET_CLASS == 1:
        threshold = THRESHOLD_1
    elif TARGET_CLASS == 2:
        threshold = THRESHOLD_2
    else:
        threshold = THRESHOLD_3

    target_contour, _ = cv.findContours(current_input, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0.0
    total_area = 0.0
    for current_contour in target_contour:
        current_area = cv.contourArea(current_contour)
        max_area = max(max_area, current_area)
        if current_area > 50:
            total_area += current_area
    if max_area > threshold:

        print(current_file + " max area is " + str(max_area) + ", total area is " + str(total_area))

        current_output_file = os.path.join(OUTPUT_PATH, current_file)
        cv.imwrite(current_output_file, current_input * 255)
    image_list.append(current_file)
    area_list.append(total_area)

result_data_frame = pd.DataFrame({CSV_ID_IMAGE: image_list, CSV_ID_AREA: area_list})
result_data_frame.to_csv(OUTPUT_AREA_CSV, header=True, index=False)
