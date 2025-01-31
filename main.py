import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from config import ht_img, wd_img, epochs, batch_size
from load_image import load_images
from process_image import process_image
from unet_model import unet
from train_step import Training
from tensorflow.keras.optimizers import Adam
from train_log import learning_curve

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
def main():
#-----------------------check GPU---------------------------
    gpus = tf.config.list_physical_devices("GPU")
    print(f"available GPUs {len(gpus)}")
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
#----------------------use all GPU-------------------------
    strategy = tf.distribute.MirroredStrategy()
    #strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
    print(f"number of devices: {strategy.num_replicas_in_sync}")
    with strategy.scope():
        images= load_images()  # load images
        images = process_image(images)  # process images
        x_train, x_val = train_test_split(images, test_size=0.2, random_state=42)  # split dataset 80% for training 20% for validation
        vae_model = Training(input_shape=(512, 512, 3), latent_dim= 256)
        vae_model.compile(optimizer=Adam(learning_rate=1e-4))
        history = vae_model.fit(
            x = x_train,
            y = None,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (x_val, None)
        )
        print("end")
        learning_curve(history) # draw learning curve
        vae_model.build(input_shape=(None, 512, 512, 3))
        vae_model.save("/home/gou/Programs/fish/result/vae_model", save_format="tf")
        print("model weight has been saved")
if __name__ == '__main__':
    main()

