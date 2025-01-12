from unet_model import unet
import tensorflow as tf


class VAEModel(tf.keras.Model):
    def __init__(self,input_shape = (512, 512 ,3), latent_dim = 256):
        super(VAEModel, self).__init__()
        self.latent_dim = latent_dim
        self.vae_unet = unet(input_shape, latent_dim)

    def compile(self, optimizer, **kwargs):
        super(VAEModel, self).compile(**kwargs)
        self.optimizer = optimizer

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = "reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = "kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name = "total_loss")

    @property
    def metrics(self):

        return [self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.total_loss_tracker]
    def vae_loss(self, y_true, y_pred, z_mean, z_log_var):
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = mse(y_true, y_pred)

        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss
        return reconstruction_loss, kl_loss, total_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data

        with tf.GradientTape() as tape:

            y_pred, z_mean, z_log_var = self.vae_unet(x, training=True)

            reconstruction_loss, kl_loss, total_loss = self.vae_loss(x, y_pred, z_mean, z_log_var)

        grads = tape.gradient(total_loss, self.vae_unet.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vae_unet.trainable_variables))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss": self.total_loss_tracker.result()
        }
    def val_step(self, data):
        #
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data

        y_pred, z_mean, z_log_var = self.vae_unet(x, training=False)
        reconstruction_loss, kl_loss, total_loss = self.vae_loss(x, y_pred, z_mean, z_log_var)

        # 更新指标
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss": self.total_loss_tracker.result()
        }
