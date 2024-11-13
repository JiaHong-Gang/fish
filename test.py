import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
#import pandas as pd
#from resnet_model import ResNet_18

from tensorflow.python.keras.combinations import generate
""""
image_path = "/Users/gangjiahong/Downloads/data1/3.県品評会/第14回千葉県若鯉品評会/1.15部総合優勝.jpg"
img = cv2.imread(image_path)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_list = cv2.calcHist([hsv_img], [0], None, [180], [0,180])
s_list = cv2.calcHist([hsv_img], [1], None, [256], [0,256])
v_list = cv2.calcHist([hsv_img], [2], None, [256], [0,256])

plt.figure(figsize = (12,4))

plt.subplot(1, 3, 1)
plt.title("hue histogram")
plt.plot(h_list, color ="orange")
plt.xlabel("hue")
plt.ylabel("frequency")

plt.subplot(1, 3, 2)
plt.title("saturation histogram")
plt.plot(s_list, color ="blue")
plt.xlabel("saturation")

plt.subplot(1, 3, 3)
plt.title("value histogram")
plt.plot(v_list, color ="green")
plt.xlabel("value")

plt.tight_layout()
plt.show()
"""
hue = (97, 110)
saturation = (110, 150)
value = (220, 255)
num_images = 10
generate_images = []
for i in range(num_images):
    hue_value = np.random.randint(hue[0],hue[1])
    saturation_value = np.random.randint(saturation[0], saturation[1])
    value_value= np.random.randint(value[0], value[1])

    hsv_img = np.zeros((224, 224, 3), dtype = np.uint8)
    hsv_img[:, :] = [hue_value, saturation_value, value_value]

    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    generate_images.append(rgb_img)
cols = 5
rows = (num_images + cols - 1) // cols
fig ,axes = plt.subplots(rows, cols, figsize = (12, 4))
fig.suptitle("Yellow images",fontsize = 16)

for i, ax in enumerate(axes.flat):
    if i < num_images:
        ax.imshow(generate_images[i])
        ax.set_title(f"image{i + 1 }")
    ax.axis("off")
plt.tight_layout()
plt.show()
""""
model = ResNet_18()
model.load_weights("/home/gou/Programs/fish/result/model_weight.h5")
csv_path = "/home/gou/Programs/fish/result/predict_blue.csv"
result = []
def add_batch(img):
    img_normalized = img / 255.0
    img_batch = np.expand_dims(img_normalized,axis = 0)
    return img_batch
for i, img in enumerate(generate_images):
    img_batch = add_batch(img)
    prediction = model.predict(img_batch)[0]
    predicted_bad = prediction[0]
    predicted_good = prediction[1]
    result.append({
        "Sample":i + 1,
        "prediction_bad_label": predicted_bad, "prediction_good_label": predicted_good
    })
    print(f"Sample {i + 1}:prediction_bad_label = {predicted_bad}, prediction_good_label = {predicted_good}")
df = pd.DataFrame(result)
df.to_csv(csv_path, index = False)
print(f"data saved in {csv_path}")
"""


