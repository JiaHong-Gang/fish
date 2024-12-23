import tensorflow as tf
from config import batch_size, epochs, ht_img, wd_img
from metric import iou_metric

from tensorflow.keras.optimizers import Adam
def train_model(x_train, x_val, y_train, y_val, model):
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    train_data = train_data.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # compile model
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy", iou_metric])
    history = model.fit(
        train_data,
        batch_size=batch_size, epochs=epochs,
        verbose=1, validation_data = val_data, shuffle=True
    )
    model.save_weights("/home/gou/Programs/fish/result/model_weight.h5")
    return  history