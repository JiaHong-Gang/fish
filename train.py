import tensorflow as tf
import numpy as np
from config import batch_size, epochs
from tensorflow.keras.optimizers import Adam

def vae_loss(y_true, y_pred, z_mean, z_log_var, beta = 1.0):
    reconstruction_loss = tf.reduce_mean(
    tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2,3])
)
    kl_per_sample = -0.5 * tf.reduce_sum(
        1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1
    )
    kl_loss = tf.reduce_mean(kl_per_sample)
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss
def get_beta(epoch, max_beta = 0.0, warmup_epochs = 100):
    return max_beta *min (1.0, (epoch + 1) / warmup_epochs)
def train_step(x_batch, model, optimizer, beta):
    with tf.GradientTape() as tape:
        outputs = model(x_batch, training=True)
        y_pred, z_mean, z_log_var = outputs
        loss, reco_loss, kl_loss = vae_loss(x_batch, y_pred, z_mean, z_log_var, beta)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, reco_loss, kl_loss
def val_step(x_batch_val, model, beta):
    outputs = model(x_batch_val, training=False)
    y_pred, z_mean, z_log_var = outputs
    loss, reco_loss, kl_loss = vae_loss(x_batch_val, y_pred, z_mean, z_log_var, beta)
    return loss, reco_loss, kl_loss
def to_numpy(val):
    try:
        return val.numpy()
    except AttributeError:
        return val
def train_model(x_train,x_val, model):
    #train dataset 
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)
    # validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.batch(batch_size)
    optimizer = Adam(learning_rate=0.0001)

    # train history
    train_losses = []
    train_reco_losses = []
    train_kl_losses = []
    val_losses = []
    val_reco_losses = []
    val_kl_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1} / epochs")
        beta = 1.0
        train_loss = 0.0
        train_reco_loss = 0.0
        train_kl_loss = 0.0
        val_loss = 0.0
        val_reco_loss = 0.0
        val_kl_loss = 0.0
        # Train loop
        n_batches = 0
        n_val_batches = 0
        for x_batch in train_dataset:
            loss, reco_loss, kl_loss = train_step(x_batch, model, optimizer, beta)
            train_loss += to_numpy(loss)
            train_reco_loss += to_numpy(reco_loss)
            train_kl_loss += to_numpy(kl_loss)
            n_batches += 1

        # Validation loop
        for x_batch_val in val_dataset:
            loss, reco_loss, kl_loss = val_step(x_batch_val, model, beta)
            val_loss += to_numpy(loss)
            val_reco_loss += to_numpy(reco_loss)
            val_kl_loss += to_numpy(kl_loss)
            n_val_batches += 1
       # Calculate means and store
        train_losses.append(train_loss / n_batches)
        train_reco_losses.append(train_reco_loss / n_batches)
        train_kl_losses.append(train_kl_loss / n_batches)
        val_losses.append(val_loss / n_val_batches)
        val_reco_losses.append(val_reco_loss / n_val_batches)
        val_kl_losses.append(val_kl_loss / n_val_batches)

        print(f"Training loss: {train_losses[-1]:.8f}, Reconstruction loss: {train_reco_losses[-1]:.8f}, KL loss: {train_kl_losses[-1]:.8f}")
        print(f"Validation loss: {val_losses[-1]:.8f}, Reconstruction loss: {val_reco_losses[-1]:.8f}, KL loss: {val_kl_losses[-1]:.8f}")

    model.save("/home/gang/programs/fish/result/model.h5")
    return train_losses, train_reco_losses, train_kl_losses, val_losses, val_reco_losses, val_kl_losses
