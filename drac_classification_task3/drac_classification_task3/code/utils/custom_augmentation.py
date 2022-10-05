import os
import cv2 as cv
import numpy as np
from PIL import Image
from scipy.stats import truncnorm
from pathlib import Path

def get_uniform_random(low=0, high=1, size=1000):
    return np.random.uniform(low=low, high=high, size=size)

def get_range_random(low=0, high=1, size=1000):
    mean = 0.5 * (low + high)
    sd = abs(high-low)/5

    return truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd).rvs(size)

def get_half_random(low=0, high=1, size=1000):
    mean = low
    sd = abs(high - low) / 4

    return truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd).rvs(size)

def gamma_correction(src, gamma):
    #gamma는 0 ~ 2
    gamma = min(2, max(0, gamma))
    if gamma < 1:
        gamma = 1 / (2 - gamma)
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)

def flip_vertical(input_image, value):
    value = min(1, max(0, value))
    output_image = input_image
    if value >= 0.5:
        output_image = cv.flip(input_image, 0)

    return output_image

def flip_horizontal(input_image, value):
    value = min(1, max(0, value))
    output_image = input_image
    if value >= 0.5:
        output_image = cv.flip(input_image, 1)

    return output_image

def rotate_image(src, angle, center_x, center_y, is_image=True):
    angle = min(90, max(-90, angle))

    image_PIL = Image.fromarray(src)
    if is_image:
        output_PIL = image_PIL.rotate(angle, resample=Image.BILINEAR, expand=False, center=(center_x, center_y))
    else:
        output_PIL = image_PIL.rotate(angle, resample=Image.NEAREST, expand=False, center=(center_x, center_y))

    output = np.array(output_PIL)
    return output

def rotate_image_on_center(src, angle, is_image=True):
    angle = min(90, max(-90, angle))

    center_x = int(0.5 * src.shape[1] + 0.5)
    center_y = int(0.5 * src.shape[0] + 0.5)

    return rotate_image(src, angle, center_x, center_y, is_image)

def bilateral_filter(input_image, value):
    value = min(100, max(0, value))
    return cv.bilateralFilter(input_image, -1, value, value)

def shapren_filter(input_image, value):
    value = min(1, max(0, value))
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv.filter2D(src=input_image, ddepth=-1, kernel=kernel)
    output_image = cv.addWeighted(input_image, (1 - value), image_sharp, value, 0)
    return output_image

def zoom_out_sliding(input_image, value, position_x=-1, position_y=-1, is_image=True):
    value = max(1, value)
    output_image = np.zeros(input_image.shape, dtype=np.uint8)
    width_input = input_image.shape[1]
    height_input = input_image.shape[0]

    if is_image:
        zoom_image = cv.resize(input_image, (0, 0), fx=1/value, fy=1/value, interpolation=cv.INTER_LINEAR)
    else:
        zoom_image = cv.resize(input_image, (0, 0), fx=1 / value, fy=1 / value, interpolation=cv.INTER_NEAREST)
    width_zoom = zoom_image.shape[1]
    height_zoom = zoom_image.shape[0]

    if position_x < 0:
        if width_input == width_zoom:
            position_x = 0
        else:
            position_x = np.random.randint(0, (width_input - width_zoom))
    if position_y < 0:
        if height_input == height_zoom:
            position_y = 0
        else:
            position_y = np.random.randint(0, (height_input - height_zoom))

    output_image[position_y:position_y + height_zoom, position_x:position_x + width_zoom] = zoom_image

    return output_image, position_x, position_y

def zoom_in_position(input_image, value, position_x, position_y, is_image=True):
    value = max(1, value)
    width_input = input_image.shape[1]
    height_input = input_image.shape[0]
    width_input_half = int(0.5 * width_input + 0.5)
    height_input_half = int(0.5 * height_input + 0.5)

    if is_image:
        zoom_image = cv.resize(input_image, (0, 0), fx=value, fy=value, interpolation=cv.INTER_LINEAR)
    else:
        zoom_image = cv.resize(input_image, (0, 0), fx=value, fy=value, interpolation=cv.INTER_NEAREST)

    width_zoom = zoom_image.shape[1]
    height_zoom = zoom_image.shape[0]

    zoom_position_x = int(position_x * value + 0.5)
    zoom_position_y = int(position_y * value + 0.5)

    min_x = max(0, zoom_position_x - width_input_half)
    min_y = max(0, zoom_position_y - height_input_half)
    max_x = min(width_zoom, min_x + width_input)
    max_y = min(height_zoom, min_y + height_input)
    if max_x == width_zoom:
        min_x = max_x - width_input
    if max_y == height_zoom:
        min_y = max_y - height_input
    output_image = zoom_image[min_y : max_y, min_x : max_x]

    return output_image

def calculate_roi_from_contour(input_contour, width, height):
    x, y, w, h = cv.boundingRect(input_contour)
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    return [x_min, y_min, x_max, y_max]

def zoom_in_center(input_image, value, is_image=True):
    width_input = input_image.shape[1]
    height_input = input_image.shape[0]

    point_center_x = int(0.5 * width_input + 0.5)
    point_center_y = int(0.5 * height_input + 0.5)

    return zoom_in_position(input_image, value, point_center_x, point_center_y, is_image)

def find_largest_external_contour(input_contours):
    max_area = 0
    result_contour = []
    for current_contour in input_contours:
        current_area = cv.contourArea(current_contour)
        if (current_area >= max_area):
            max_area = current_area
            result_contour = current_contour
    return result_contour

def get_segmentation_center(label_image):
    contours, _ = cv.findContours(label_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = find_largest_external_contour(contours)
    M = cv.moments(largest_contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    return center_x, center_y

def change_path(original_path, from_name, to_name):
    output_name = original_path.replace(from_name, to_name)
    path = Path(output_name).parent
    if not os.path.exists(path):
        os.makedirs(path)

    return output_name
