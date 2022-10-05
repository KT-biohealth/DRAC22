import os
import numpy as np
import cv2 as cv

RESULT1_FOLDER = r"./result/segformer_mit-b5_80k_drac2022seg" # test result 1 path
RESULT2_FOLDER = r"./result/upernet_convnext_xlarge_80k_drac2022seg" # test result 2 path

TARGET_NUM = "2"

MIN_BG_AREA = 3000 #delete small background area
MIN_AREA = 1000 #delete small area

KERNEL_SIZE = 7
input_result1_folder = os.path.join(RESULT1_FOLDER, r"MICCAI", TARGET_NUM)
input_result2_folder = os.path.join(RESULT2_FOLDER, r"MICCAI", TARGET_NUM)
output_folder = os.path.join(RESULT1_FOLDER, r"MICCAI_sum", TARGET_NUM)

def delete_small_Area(img, min_area):

    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(img)

    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs = nb_blobs - 1

    result_img = np.zeros((img.shape), np.uint8)

    for blob in range(nb_blobs):
        if sizes[blob] >= min_area:
            result_img[im_with_separated_blobs == blob + 1] = 255

    return result_img

if __name__ == '__main__':

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    file_list = os.listdir(input_result1_folder)
    for current_file in file_list:

        current_output_file = os.path.join(output_folder, current_file)
        current_result1_image = cv.imread(os.path.join(input_result1_folder, current_file), cv.IMREAD_GRAYSCALE)
        current_result1_image = cv.morphologyEx(current_result1_image, cv.MORPH_CLOSE, kernel)

        current_result2_image = cv.imread(os.path.join(input_result2_folder, current_file), cv.IMREAD_GRAYSCALE)
        current_result2_image = cv.morphologyEx(current_result2_image, cv.MORPH_CLOSE, kernel)

        output_image = current_result1_image & current_result2_image # logical OR
        output_image = cv.morphologyEx(output_image, cv.MORPH_CLOSE, kernel)

        output_image = np.invert(delete_small_Area(np.invert(output_image * 255), MIN_BG_AREA))
        output_image = delete_small_Area(output_image, MIN_AREA)
        output_image = np.clip(output_image, 0, 1)

        cv.imwrite(current_output_file, output_image)

