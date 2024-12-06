import numpy as np
import cv2
from loadimage import load_image_and_mask
from config import ht_img, wd_img
def process_image(images):
    pro_img = []
    for img in images:
        resized_img = cv2.resize(img,(ht_img, wd_img)) # resize image to 1024x1024
        resized_img = resized_img/255.0  # nomorazate image
        pro_img.append(resized_img)
    images = np.array(pro_img) 
    print("image process finished")
    return images

