from vae_model import vae
import tensorflow as tf

class VAEModel(tf.keras.Model):
    def __init__(self,input_shape = (1088, 768 ,3), latent_dim = 256):
        super(VAEModel, self).__init__()
        self.latent_dim = latent_dim
        self.vae = vae(input_shape, latent_dim)
        
    def call(self, inputs, training=None):
        return self.vae(inputs, training=training)

    def compile(self, optimizer, **kwargs):
        super(VAEModel, self).compile(optimizer = optimizer, **kwargs)

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = "reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = "kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name = "total_loss")

    @property
    def metrics(self):

        return [self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.total_loss_tracker]
    def vae_loss(self, y_true, y_pred, z_mean, z_log_var):
        per_sample_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis = [1, 2, 3])
        reconstruction_loss = tf.reduce_mean(per_sample_mse)

        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss
        return reconstruction_loss, kl_loss, total_loss
