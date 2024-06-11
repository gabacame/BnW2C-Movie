import cv2
import numpy as np

def convert_to_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = l / 255.0
    a = (a - 128) / 128.0
    b = (b - 128) / 128.0
    return l, a, b
