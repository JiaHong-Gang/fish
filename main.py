import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from config import ht_img, wd_img,epochs, batch_size, num_class
from load_image import load_images
from process_image import process_image
from pair_image import pair
from unet_model import unet
from metric import iou_metric
from train_log import plot_curve, save_train_log
from test import predictions
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
def main():
#-----------------------check GPU---------------------------
    gpus = tf.config.list_physical_devices("GPU")
    print(f"available GPUs {len(gpus)}")
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
#----------------------use all GPU-------------------------
    #strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
    print(f"number of devices: {strategy.num_replicas_in_sync}")
    print("Is GPU available:", tf.config.list_logical_devices('GPU'))
    tf.config.optimizer.set_jit(False)
    with (strategy.scope()):
        images= load_images(img_folder = "/home/gang/fish/IDdata", is_mask = False)  # load images
        body_shape = load_images(img_folder ="/home/gang/fish/masks", is_mask = True  ) # load body shape mask
        red_area = load_images(img_folder ="/home/gang/fish/red_mask", is_mask = True ) # load red area mask
        white_area = load_images(img_folder= "/home/gang/fish/white_mask", is_mask = True) # load white area mask
        images = process_image(images, is_mask = False)  # process images
        body_shape = process_image(body_shape, is_mask = True) # process body shape mask
        red_area = process_image(red_area, is_mask = True) # process red area mask
        white_area = process_image(white_area, is_mask = True) # process white area mask
        train_data, train_mask, val_data, val_mask = pair(images, body_shape, red_area, white_area)# split dataset 80% for training 20% for validation
        model= unet(input_shape=(ht_img, wd_img, 3))  # use unet model
        model.summary()
        model.compile(optimizer = Adam(learning_rate = 1e-4),loss = "binary_crossentropy", metrics = [iou_metric])
        history =model.fit(
            x = train_data,
            y = train_mask,
            validation_data = (val_data, val_mask), 
            batch_size = batch_size,
            epochs = epochs,
        )
        model.save("/home/gang/programs/fish/result/model.h5")
        plot_curve(history)
        save_train_log(history)
        predictions(model, val_data, val_mask)
if __name__ == '__main__':
    main()

