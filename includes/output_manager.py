#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import cv2
import copy
import os

class Display():
    def __init__(self):
        pass

    @staticmethod
    def _show_image(image, window_name="", window_size=None, wait_sec=10):
        if window_size is None:
            window_size = (1280,720)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_size[0], window_size[1])
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_sec)
