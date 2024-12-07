import numpy as np
import cv2
from loadimage import load_image_and_mask
from config import ht_img, wd_img
def process_image(images, is_mask = False):
    pro_img = []
    for img in images:
        resized_img = cv2.resize(img,(ht_img, wd_img)) # resize image to 1024x1024
        if not is_mask:
            resized_img = resized_img/255.0  # normalized image
        else:
            resized_img = resized_img / 255.0  # normalized image
            resized_img = np.expand_dims(resized_img, axis = -1) #add channel dimension
        pro_img.append(resized_img)
    images = np.array(pro_img)
    if is_mask:
        print("mask process finished")
    else:
        print("image process finished")
    return images

