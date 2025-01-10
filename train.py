import tensorflow as tf
from config import batch_size, epochs
from tensorflow.keras.optimizers import Adam

def vae_loss(y_ture, y_pred, z_mean, z_log_var):
    mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.NONE)
    reconstruction_loss = tf.reduce_mean(mse(y_ture, y_pred))
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
def val_step(x_batch_val, model):
    y_pred, z_mean, z_log_var = model(x_batch_val, training = False)
    loss = vae_loss(x_batch_val, y_pred, z_mean, z_log_var)
    return loss
def train_model(x_train,x_val, model, strategy):
    #train dataset
    with strategy.scope():
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        # validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = strategy.experimental_distribute_dataset(val_dataset)
        optimizer = Adam(learning_rate=0.0001)
    # train history
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1} / epochs")
        train_loss = 0.0
        val_loss = 0.0
        for step, x_batch_train in enumerate(train_dataset):
            per_replica_loss = strategy.run(train_step, args=(x_batch_train, model, optimizer))
            step_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            train_loss += step_loss.numpy()
        for x_batch_val in val_dataset:
            per_replica_val_loss = strategy.run(val_step, args=(x_batch_val,model))
            val_step_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_loss, axis=None)
            val_loss += val_step_loss.numpy()
        #calculate mean and print
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        print(f"Training loss : {epoch_train_loss:.4f}")
        print(f"Validation loss : {epoch_val_loss:.4f}")

    model.save("/home/gou/Programs/fish/result/model.h5")
    return train_losses, val_losses