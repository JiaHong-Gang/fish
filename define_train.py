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
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        reconstruction_loss = mse(y_true, y_pred)

        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss
        return reconstruction_loss, kl_loss, total_loss
