import numpy as np
import cv2
from config import ht_img, wd_img
def process_image(images, is_mask = False):
    pro_img = []
    for img in images:
        if is_mask == False:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (wd_img, ht_img))  # resize image to 1088x768
        resized_img = resized_img / 255.0  # normalized image
        pro_img.append(resized_img)
    images = np.array(pro_img)
    print(f"Processing of {len(images)} images has been completed")
    return images