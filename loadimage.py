import cv2
import os

def load_image_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file == ".DS_Store":
                continue
            image_path = os.path.join(root, file)
            if os.path.isfile(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                else:
                    print("error!, Can't read image from image_path")
    return images

def load_image():
    folder1 = "/home/gou/fish/data1"
    folder2 = "/home/gou/fish/data2"

    image1 = load_image_from_folder(folder1)
    image2 = load_image_from_folder(folder2)
    images = image1 + image2
    print ("total images are ",len(images))
    return images


