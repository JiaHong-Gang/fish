import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Memory growth setting failed:", e)
from config import ht_img, wd_img, epochs, batch_size
from load_image import load_images
from process_image import process_image
from pair_image import pair
from vae_model import vae
from train_step import Training
from train_log import learning_curve, training_log

def main():
#-----------------------check GPU---------------------------
    print(f"available GPUs {len(gpus)}")
    for gpu in gpus:
        print(gpu)
    print("Is GPU available:", tf.config.list_logical_devices('GPU'))
#----------------------programs-------------------------
    images = load_images("/home/gang/fish/IDdata", False)  # load images
    mask = load_images("home/gang/fish/mask", True)
    images = process_image(images, False)# process images
    mask = process_image(mask, True)
    x_train, x_val= pair(images, mask)  # split dataset 80% for training 20% for validation
    vae_model = Training(input_shape=(ht_img, wd_img, 3), latent_dim= 256)
    vae_model.compile(optimizer=Adam(learning_rate=1e-4))
    history = vae_model.fit(
        x = x_train,
        y = None,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (x_val, None)
    )
    print("end")

    sample = x_val[:5]
    outputs = vae_model(sample, training = False)
    outputs = outputs[0]
    for i , output in enumerate(outputs):
        print(f"Image {i}: min={np.min(output):.4f}, max={np.max(output):.4f}, mean={np.mean(output):.4f}, std={np.std(output):.4f}")
    training_log(history)# save training log
    learning_curve(history) # draw learning curve
    #vae_model.build(input_shape=(None, 1088, 768, 3))
    vae_model.save("/home/gang/programs/fish/result/vae_model", save_format="tf")
    print("model weight has been saved")
if __name__ == '__main__':
    main()

