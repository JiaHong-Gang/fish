import cv2
import os
import sys
def load_images():
    images = []
    img_folder = "/home/gou/fish/IDdata"
    #img_folder = "/Users/gangjiahong/Downloads/IDdata"
    if not os.path.exists(img_folder):
        print(f"Error, folder {img_folder} dose not exist.")
        sys.exit(1)
    # load images
    for img_file in os.listdir(img_folder):
        if img_file.startswith('.'): #ignore . file
            continue
        img_path = os.path.join(img_folder, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    print(f"{len(images)} images has been loaded.")
    return images