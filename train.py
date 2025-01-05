import tensorflow as tf
from config import batch_size, epochs, ht_img, wd_img
from add_noisy import map_func
from tensorflow.keras.optimizers import Adam

def train_model(x_train,x_val, model):
    #tarin dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size = 1000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(map_func, num_parallel_calls = tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(map_func, num_parallel_calls = tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    # compile model
    model.compile(loss="MSE", optimizer=Adam(learning_rate=0.0001))
    history = model.fit(
        train_dataset,
        epochs=epochs,
        verbose=1,
        validation_data = val_dataset
    )
    model.save("/home/gou/Programs/fish/result/model.h5")
    return  history