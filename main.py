import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from config import ht_img, wd_img
from loadimage import load_image_and_mask
from imageprocess import process_image
from unet_model import unet
from train import train_model
from trainlog import plot_curve
from predict import prediction, predict_block_image, feature_dim_reduce

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
    print(f"number of devices: {strategy.num_replicas_in_sync}")
    with strategy.scope():
        images, mask = load_image_and_mask()  # load images and masks
        images = process_image(images, is_mask=False)  # process images
        masks = process_image(mask, is_mask=True)  # process masks
        x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2,
        random_state=42)  # split dataset 80% for training 20% for validation
        model = unet(input_shape=(ht_img, wd_img, 3))  # use unet model
        #history = train_model(x_train, x_val, y_train, y_val, model)
        #plot_curve(history) # draw learning curve
        #prediction(x_val, y_val,model)
        #predict_block_image(x_val, y_val, model, 256)
        feature_dim_reduce(x_val, y_val, model)
if __name__ == '__main__':
    main()

