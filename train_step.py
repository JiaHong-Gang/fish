import tensorflow as tf
from define_train import VAEModel

class Training(VAEModel):
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
    def test_step(self, data):
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
