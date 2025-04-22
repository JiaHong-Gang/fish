import tensorflow as tf
from config import batch_size, epochs
from tensorflow.keras.optimizers import Adam
from config import alpha
from perceptual import compute_perceptual_loss
def vae_loss(y_ture, y_pred, z_mean, z_log_var):
    mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.NONE)
    reconstruction_loss = tf.reduce_mean(mse(y_ture, y_pred))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    perceptual_loss = compute_perceptual_loss(y_ture, y_pred)
    total_loss = reconstruction_loss + kl_loss + alpha * perceptual_loss
    return total_loss, reconstruction_loss, kl_loss, perceptual_loss

def train_step(x_batch, model, optimizer):
    with tf.GradientTape() as tape:
        y_pred, z_mean, z_log_var = model(x_batch, training = True)
        loss, reco_loss, kl_loss, perceptual_loss = vae_loss(x_batch, y_pred, z_mean, z_log_var)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, reco_loss, kl_loss, perceptual_loss
def val_step(x_batch_val, model):
    y_pred, z_mean, z_log_var = model(x_batch_val, training = False)
    loss, reco_loss, kl_loss, perceptual_loss = vae_loss(x_batch_val, y_pred, z_mean, z_log_var)
    return loss, reco_loss, kl_loss, perceptual_loss
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
    #calulate batch number
    num_train_batch = len(x_train) // batch_size
    if len(x_train) % batch_size != 0:
        num_train_batch += 1
    num_val_batch = len(x_val) // batch_size
    if len(x_val) %batch_size != 0:
        num_val_batch +=1
    # train history
    train_losses = []
    train_reco_losses = []
    train_kl_losses = []
    train_perceptual_losses = []
    val_losses = []
    val_reco_losses = []
    val_kl_losses = []
    val_perceptual_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1} / epochs")
        train_loss = 0.0
        train_reco_loss = 0.0
        train_kl_loss = 0.0
        train_perceptual_loss = 0.0
        val_loss = 0.0
        val_reco_loss = 0.0
        val_kl_loss = 0.0
        val_perceptual_loss = 0.0
        for step, x_batch_train in enumerate(train_dataset):
            per_replica_losses = strategy.run(train_step, args=(x_batch_train, model, optimizer))
            step_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            step_reco_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
            step_kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None)
            step_perceptual_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[3], axis = None)
            train_loss += step_loss.numpy()
            train_reco_loss += step_reco_loss.numpy()
            train_kl_loss += step_kl_loss.numpy()
            train_perceptual_loss += step_perceptual_loss.numpy()
        
        # Validation loop
        for x_batch_val in val_dataset:
            per_replica_val_losses = strategy.run(val_step, args=(x_batch_val, model))
            val_step_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses[0], axis=None)
            val_step_reco_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses[1], axis=None)
            val_step_kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses[2], axis=None)
            val_step_perceptual_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses[3], axis=None)
            val_loss += val_step_loss.numpy()
            val_reco_loss += val_step_reco_loss.numpy()
            val_kl_loss += val_step_kl_loss.numpy()
            val_perceptual_loss += val_step_perceptual_loss.numpy()
        
        # Calculate means and store
        epoch_train_loss = train_loss / num_train_batch
        epoch_train_reco_loss = train_reco_loss / num_train_batch
        epoch_train_kl_loss = train_kl_loss / num_train_batch
        epoch_train_perceptual_loss = train_perceptual_loss / num_train_batch
        train_losses.append(epoch_train_loss)
        train_reco_losses.append(epoch_train_reco_loss)
        train_kl_losses.append(epoch_train_kl_loss)
        train_perceptual_losses.append(epoch_train_perceptual_loss)
        
        epoch_val_loss = val_loss / num_val_batch
        epoch_val_reco_loss = val_reco_loss / num_val_batch
        epoch_val_kl_loss = val_kl_loss / num_val_batch
        epoch_val_perceptual_loss = val_perceptual_loss / num_val_batch
        val_losses.append(epoch_val_loss)
        val_reco_losses.append(epoch_val_reco_loss)
        val_kl_losses.append(epoch_val_kl_loss)
        val_perceptual_losses.append(epoch_val_perceptual_loss)
        
        print(f"Training loss: {epoch_train_loss:.8f}, Reconstruction loss: {epoch_train_reco_loss:.8f}, KL loss: {epoch_train_kl_loss:.8f}, perceptual loss: {epoch_train_perceptual_loss:.8f}")
        print(f"Validation loss: {epoch_val_loss:.8f}, Reconstruction loss: {epoch_val_reco_loss:.8f}, KL loss: {epoch_val_kl_loss:.8f}, perceptual loss: {epoch_val_perceptual_loss:.8f}")
    
    model.save("/home/gou/Programs/fish/result/model.h5")
    return train_losses, train_reco_losses, train_kl_losses, train_perceptual_losses,  val_losses, val_reco_losses, val_kl_losses, val_perceptual_losses