from sklearn.model_selection import train_test_split
import numpy as np

def pair(images, body_shape):
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
    x_train = {
        "input_image": train_images,
        "mask":train_body_shape
    }
    x_val = {
        "input_image": val_images,
        "mask": val_body_shape
    }
    return  x_train, x_val