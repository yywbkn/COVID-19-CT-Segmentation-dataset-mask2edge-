import os
import shutil
# from libtiff import TIFF
from scipy import misc
import random
import cv2
import numpy as np




def binary2edge(mask_path):
    """
    func1: threshold(src, thresh, maxval, type[, dst]) -> retval, dst
            https://www.cnblogs.com/FHC1994/p/9125570.html
    func2: Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges

    :param mask_path:
    :return:
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ret, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)  # if <0, pixel=0 else >0, pixel=255
    mask_edge = cv2.Canny(mask_binary, 10, 150)

    return mask_edge


def binaryMask(im_path):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    ret, mask_binary = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)

    return mask_binary


if __name__ == '__main__':


    #
    path = './TestingSet/LungInfection-Test/GT'
    save = './Edge'

    for img_name in os.listdir(path):
        img_path = path + '/' + img_name
        print('img_path =',img_path)
        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_edge = cv2.Canny(im, 10, 150)
        save_path = save +'/' + img_name
        cv2.imwrite(save_path, mask_edge)
        print(mask_edge.shape)




