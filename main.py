import tensorflow as tf
import os
from train import train_model
from trainlog import plot_curve
from predict import prediction
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
def main():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"available GPUs {len(gpus)}")
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    print(f"number of devices: {strategy.num_replicas_in_sync}")
    with strategy.scope():
        model, history, x_val, y_val = train_model()
        plot_curve(history)
        prediction(x_val, y_val,model)
if __name__ == '__main__':
    main()

