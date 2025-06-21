from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

def pair(images, body_shape, batch_size = 4, shuffle_buffer = 256):
    train_merged = []
    val_merged = []
    data = list(zip(images, body_shape))
    train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 42)
    train_images, train_body_shape = zip(*train_data)
    val_images, val_body_shape = zip(*test_data)
    train_images = np.array(train_images)
    train_body_shape = np.array(train_body_shape)
    val_images = np.array(val_images)
    val_body_shape = np.array(val_body_shape)
    train_images = train_images.astype(np.float32)
    train_body_shape  = train_body_shape.astype(np.float32)
    val_images   = val_images.astype(np.float32)
    val_body_shape    = val_body_shape.astype(np.float32)
    x_train = tf.data.Dataset.from_tensor_slices({
        "input_image": train_images,
        "mask": train_body_shape
    })
    x_val = tf.data.Dataset.from_tensor_slices({
        "input_image": val_images,
        "mask": val_body_shape
    })
    x_train = x_train.shuffle(shuffle_buffer) \
                       .batch(batch_size) \
                       .prefetch(tf.data.AUTOTUNE)
    x_val   = x_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return  x_train, x_val
