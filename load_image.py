import cv2
import os
import sys
def load_images(img_folder = "/home/gang/fish/IDdata"):
    images = []
    total = 0
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
            total += 1
            print(f"loaded {total} images...",end="\r")
        else:
            print(f"warning: failed to load image{img_path}")
    print(f"{total} images has been loaded.")
    return images