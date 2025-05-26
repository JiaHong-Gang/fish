import numpy as np
import cv2
from config import ht_img, wd_img
def process_image(images):
    pro_img = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img,(wd_img, ht_img)) # resize image to 720x512
        resized_img = resized_img/255.0  # normalized image
        pro_img.append(resized_img)
    images = np.array(pro_img)
    print(f"Processing of {len(images)} images has been completed")
    return images