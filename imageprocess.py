import numpy as np
import cv2
from loadimage import load_image
from config import ht_img, wd_img, soft_label
def process_image():
    images = load_image()
    pro_img = []
    for img in images:
        resized_img = cv2.resize(img,(ht_img, wd_img))
        resized_img = resized_img/255.0
        pro_img.append(resized_img)
    images = np.array(pro_img) 
    print("image process finished")   
    return images
def label(images):
    if sum(soft_label) == 1.00:
        labels = np.array([soft_label]*len(images))
        print("label process finished")
    else :
        print("error! the sum is not 1.00")

        return None

    return labels

