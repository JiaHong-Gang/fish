import cv2
import matplotlib.pyplot as plt
from process_image import add_noisy
import tensorflow as tf
from config import ht_img,wd_img
t = 500
train_data_path = "/Users/gangjiahong/Downloads/data1/3.県品評会/第14回千葉県若鯉品評会/1.15部総合優勝.jpg"
train_data = cv2.imread(train_data_path)
train_data = cv2.cvtColor(train_data, cv2.COLOR_BGR2RGB)
train_data = cv2.resize(train_data,(ht_img, wd_img))
train_data = tf.convert_to_tensor(train_data, dtype=tf.float32) / 255.0
train_data = add_noisy(train_data, t)
train_data = tf.clip_by_value(train_data, 0.0, 1.0)
plt.figure(figsize= (10, 8))
plt.title("noisy image")
plt.imshow(train_data)
plt.show()















