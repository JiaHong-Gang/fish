import tensorflow as tf
from config import batch_size, epochs
from tensorflow.keras.optimizers import Adam

def vae_loss(y_ture, y_pred, z_mean, z_log_var):
    mse = tf.keras.losses.meanSquaredError()
    reconstruction_loss = mse(y_ture, y_pred)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    total_loss = reconstruction_loss + kl_loss
    return total_loss

def train_step(model, x_batch, optimizer):
    with tf.GradientTape() as tape:
        y_pred, z_mean, z_log_var = model(x_batch, training = True)
        loss = vae_loss(x_batch, y_pred, z_mean, z_log_var)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model(x_train,x_val, model):
    #train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size = 1000)
    train_dataset = train_dataset.batch(batch_size)
    # validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.batch(batch_size)
    optimizer = Adam(learning_rate = 0.0001)
    #train history
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1} / epochs")
        train_loss = 0.0
        val_loss = 0.0
        for step, x_batch_train in enumerate(train_dataset):
            loss = train_step(model, x_batch_train, optimizer)
            train_loss += loss.numpy()
        for x_batch_val in val_dataset:
            y_pred, z_mean, z_log_var = model(x_batch_val, training = False)
            val_loss = vae_loss(x_batch_val, y_pred, z_mean, z_log_var)
            val_loss += val_loss.numpy()
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        print(f"Training loss : {epoch_train_loss:.4f}")
        print(f"Validation loss : {epoch_val_loss:.4f}")

    model.save("/home/gou/Programs/fish/result/model.h5")
    return  model, train_losses, val_losses