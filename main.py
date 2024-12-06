from train import train_model
from trainlog import plot_curve
import tensorflow as tf
def main():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"available GPUs{len(gpus)}")
    for gpu in gpus:
        print(gpu)
    strategy = tf.distribute.MirroredStrategy()
    print(f"number of devices: {strategy.num_replicas_in_sync}")
    with strategy.scope():
        model, history, x_val, y_val = train_model()
        plot_curve(history)
if __name__ == '__main__':
    main()

