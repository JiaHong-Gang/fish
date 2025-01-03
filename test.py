import cv2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from unet_model import unet
from config import ht_img,wd_img
train_data_path = "/home/gou/fish/IDdata/ID1086_3.70部総合.JPG"
train_data = cv2.imread(train_data_path)
train_data = cv2.cvtColor(train_data, cv2.COLOR_BGR2RGB)
train_data = cv2.resize(train_data,(ht_img, wd_img))
train_data = train_data /255.0
train_data = tf.expand_dims(train_data, axis= 0)
model = unet(input_shape=(ht_img, wd_img, 3), time_dim= 1024)
model.compile(loss="MSE", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])
time_steps = tf.random.uniform((1, 1), minval=0, maxval=1000, dtype=tf.int32)
history = model.fit(
    [train_data, time_steps],train_data,
    batch_size=1, epochs=1,
    verbose=1, shuffle=True
)















