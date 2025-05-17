import cv2
import os
import sys
def load_images(img_folder = "/home/gang/fish/IDdata", is_mask = False):
    images = []
    file_list = []
    total = 0
    #img_folder = "/Users/gangjiahong/Downloads/IDdata"
    if not os.path.exists(img_folder):
        print(f"Error, folder {img_folder} dose not exist.")
        sys.exit(1)
    # load images
    for file in os.listdir(img_folder):
        if file.startswith('.'): #ignore . file
            continue
        file_list.append(file)
    file_list.sort(key = lambda name : name.split("_")[0])
    print(file_list[:10])
    for file in file_list:
        img_path = os.path.join(img_folder, file)
        if is_mask:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            total += 1
            print(f"loaded {total} images...",end="\r")
        else:
            print(f"warning: failed to load image{img_path}")
    print(f"{total} images has been loaded from {img_folder}.")
    return images