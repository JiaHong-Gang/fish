import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from config import ht_img, wd_img,epochs
from load_image import load_images
from process_image import process_image
from vae_model import vae
from train import train_model
from train_log import plot_curve,save_train_log

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
    with (strategy.scope()):
        images= load_images()  # load images
        images = process_image(images)  # process images
        x_train, x_val = train_test_split(images, test_size=0.2, random_state=42)  # split dataset 80% for training 20% for validation
        model= vae(input_shape=(ht_img, wd_img, 3))  # use unet model
        train_losses, train_reco_losses, train_kl_losses, train_perceptual_losses, val_losses, val_reco_losses, val_kl_losses ,val_perceptual_losses = train_model(x_train, x_val, model, strategy)

        save_train_log(train_losses, train_reco_losses, train_kl_losses, train_perceptual_losses, val_losses, val_reco_losses, val_kl_losses ,val_perceptual_losses)
        plot_curve(train_losses, train_reco_losses, train_kl_losses, train_perceptual_losses, val_losses, val_reco_losses, val_kl_losses ,val_perceptual_losses, epochs) # draw learning curve

if __name__ == '__main__':
    main()

