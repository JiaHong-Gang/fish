from sklearn.model_selection import train_test_split
import numpy as np

def pair(images, body_shape, red_area, white_area):
    train_merged = []
    val_merged = []
    data = list(zip(images, body_shape, red_area, white_area))
    train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 42)
    train_images, train_body_shape, train_red_area, train_white_area = zip(*train_data)
    val_images, val_body_shape, val_red_area, val_white_area = zip(*test_data)
    for m1, m2, m3, in zip(train_body_shape, train_red_area, train_white_area):
        combined1 = np.stack([m1, m2, m3], axis = -1)
        train_merged.append(combined1)
    for m1, m2, m3 in zip(val_body_shape, val_red_area, val_white_area):
        combined2 = np.stack([m1, m2, m3], axis = -1)
        val_merged.append(combined2)
    return np.array(train_images), np.array(train_merged), np.array(val_images), np.array(val_merged)

