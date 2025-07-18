import cv2
import numpy as np
import torch


def read_image(image_path:str, image_size:int=224):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    return image

def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch