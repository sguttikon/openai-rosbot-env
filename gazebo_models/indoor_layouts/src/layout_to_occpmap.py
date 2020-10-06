#!/usr/bin/env python3

import argparse
import numpy as np
import cv2 as cv

def get_occupany_map(img: np.ndarray, output_path: str, output_size: tuple):
    """
    Convert the HouseExpo indoor layout to occupancy map

    Parameters
    ----------
    img: numpy.ndarray
        input image
    output_path: str
        full path (*.pgm) indicating where to store the file
    output_size: tuple
        the output image shape as (rows, cols)
    """

    (rows, cols) = img.shape

    # the houseexpo layout has black-background so perform inverse transform
    _, thresh_img = cv.threshold(img, 1, 255, cv.THRESH_BINARY_INV)

    # find and draw contours i.e, borders
    # reference: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(thresh_img, contours, contourIdx=1, color=100, thickness=3)

    # switch colors => 205: unknown, 255: free, 0: occupied
    thresh_img = np.where(thresh_img==255, np.uint8(205), thresh_img) # unknown
    thresh_img = np.where(thresh_img==0, np.uint8(255), thresh_img) # free
    thresh_img = np.where(thresh_img==100, np.uint8(0), thresh_img) # obstacle

    # add padding to borders to make the output have equal width and height
    padding = max(rows, cols) + 50
    thresh_img = cv.copyMakeBorder(thresh_img, (padding-rows)//2, (padding-rows)//2, \
            (padding-cols)//2, (padding-cols)//2, cv.BORDER_CONSTANT, value=205)
    thresh_img = cv.resize(thresh_img, output_size)

    # store the image
    cv.imwrite(output_path, thresh_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HouseExpo indoor layout images to occupany map')
    parser.add_argument('--file_path', dest='file_path', \
                    required=True, help='full path to layout image')
    parser.add_argument('--output_path', dest='output_path', \
                    required=True, help='full path to output image (*.pgm)')
    parser.add_argument('--o_shape', dest='o_shape', type=int, \
                    default=768, help='output occupancy map shape')
    args = parser.parse_args()

    img = cv.imread(args.file_path, cv.IMREAD_UNCHANGED)
    get_occupany_map(img, args.output_path, (args.o_shape, args.o_shape))
