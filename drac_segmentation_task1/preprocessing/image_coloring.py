import cv2
import os
import numpy as np

SAVE_FOLDER = r"../data/DRAC2022_Seg_Color"

def read_list(Listtxt):

    with open(Listtxt) as f:
        Path = []

        for line in f:
            try:
                value = line[:-1]
            except ValueError:
                value = line.strip('\n')
            Path.append(value)

    return Path

def coloring(imgpaths):

    thres = 120

    for imgpath in imgpaths:

        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

        ch0 = img
        ch1 = np.zeros_like(img)
        ch2 = np.zeros_like(img)

        ch1[img > thres] = img[img > thres]
        ch2[img <= thres] = img[img <= thres]

        result_img = cv2.merge((ch0, ch1, ch2))

        cv2.imwrite(imgpath.replace('Seg', 'Seg_Color'), result_img)


if __name__ == '__main__':

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    imglist = 'drac2022seg_all.txt'
    imgpaths = read_list(imglist)

    coloring(imgpaths)
