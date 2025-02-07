import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from cv2.gapi import BGR2RGB

folder_path = "/Users/gangjiahong/Downloads/IDdata"
save_path = "/Users/gangjiahong/Desktop/rotation_image"
save_original_image_path ="/Users/gangjiahong/Desktop/original_image"
image_file = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
image = []
random.shuffle(image_file)
for idx, file in enumerate(image_file[:5]):
    img = cv2.imread(file)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image.append(img)
    plt.imsave(os.path.join(save_original_image_path, f"original_image{idx +1}.jpg"), img)

#-----------split image
def vertical_split(image):
    left_parts = []
    right_parts = []
    new_vertical_images = []
    for img in image:
        left_part = img[:, :256]
        right_part = img[:, 256:]
        left_parts.append(left_part)
        right_parts.append(right_part)
    while True:
        random.shuffle(right_parts)
        if all(right_parts[i] is left_parts[i] for i in range(len(right_parts))):
            continue
        break
    for i in range(5):
        combined_vertical_image = np.hstack((left_parts[i], right_parts[i]))
        new_vertical_images.append(combined_vertical_image)
        plt.figure(figsize=(12, 5))
        plt.imshow(new_vertical_images[i])
        plt.title("vertical split image")
        plt.axis("off")
        plt.imsave(os.path.join(save_path, f"vertical_split_image{i}.jpg"), new_vertical_images[i])
        plt.show()

def horizon_spilt(image):
    upper_parts = []
    lower_parts = []
    new_horizon_images = []
    for img in image:
        upper_part = img[256:, :]
        lower_part = img[:256, :]
        upper_parts.append(upper_part)
        lower_parts.append(lower_part)
    while True:
        random.shuffle(upper_parts)
        if all(upper_parts[i] is lower_parts[i] for i in range(len(upper_parts))):
            continue
        break
    for i in range(5):
        combined_horizon_image = np.vstack((lower_parts[i], upper_parts[i]))
        new_horizon_images.append(combined_horizon_image)
        plt.imshow(new_horizon_images[i])
        plt.title("horizon split image")
        plt.axis("off")
        plt.imsave(os.path.join(save_path, f"horizon_split_image{i}.jpg"), new_horizon_images[i])
        plt.show()
#-----------change red area

def change_red(image):
    for idx ,img in enumerate(image):
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        mask = np.logical_and.reduce([R > G, R > B, G < 100, B < 100]).astype(np.uint8) * 255
        img_result = img.copy()
        img_result[mask == 255] = [120 , 40 , 40]
        plt.figure(figsize=(12,4))
        plt.title(f"change red area")
        plt.imshow(img_result)
        plt.show()
        plt.imsave(os.path.join(save_path, f"change color{idx + 1}.jpg"), img_result)
#-------change white area
def change_white(image):
    for idx ,img in enumerate(image):
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        mask = np.logical_and.reduce([R > 150, G > 150,  B > 150]).astype(np.uint8) * 255
        img_result = img.copy()
        img_result[mask == 255] = [0 , 0 , 0]
        plt.figure(figsize=(12,4))
        plt.title(f"change white area")
        plt.imshow(img_result)
        plt.show()
        plt.imsave(os.path.join(save_path, f"change color{idx + 1}.jpg"), img_result)
def change_background(image):
    for idx, img in enumerate(image):
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        mask = np.logical_and.reduce([R< 110, G < 160, B > 200]).astype(np.uint8) * 255
        random_color = np.random.randint(0, 256, 3, dtype = np.uint8)
        img_result = img.copy()
        img_result[mask == 255] = random_color
        plt.figure(figsize= (12, 4))
        plt.title(f" change background")
        plt.axis("off")
        plt.imshow(img_result)
        plt.show()
        plt.imsave(os.path.join(save_path, f"change background{idx +1}.jpg"), img_result)
def rotation_image(image):
    for idx, img in enumerate(image):
        angle =random.choice([90, 180, 270])
        if angle ==90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle ==270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plt.figure(figsize=(12, 4))
        plt.title(f"rotation image")
        plt.imshow(img)
        plt.show()
        plt.imsave(os.path.join(save_path, f"rotation_image{idx + 1}.jpg"), img)
rotation_image(image)
