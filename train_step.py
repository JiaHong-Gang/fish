import tensorflow as tf
from define_train import VAEModel
from perceptual import compute_perceptual_loss

class Training(VAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       
    def train_step(self, data):
        x_img  = data

        with tf.GradientTape() as tape:

            y_pred,  z_mean, z_log_var = self.vae(x_img, training=True)

            reconstruction_loss, kl_loss, _ = self.vae_loss(x_img, y_pred, z_mean, z_log_var)
            perceptual_loss = compute_perceptual_loss(x_img, y_pred)         
            total_loss = (
                self.reconstruction_weight *reconstruction_loss + 
                self.kl_weight * kl_loss + 
                self.perceptual_weight * perceptual_loss
            )

        grads = tape.gradient(total_loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_variables))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
            "loss": self.total_loss_tracker.result()
        }
    def test_step(self, data):
        #
        if isinstance(data, tuple):
            x= data
        else:
            x = data
        x_img = x

        y_pred, z_mean, z_log_var = self.vae(x_img, training=False)
        reconstruction_loss, kl_loss, _ = self.vae_loss(x_img, y_pred, z_mean, z_log_var)
        perceptual_loss = compute_perceptual_loss(x_img, y_pred)
        total_loss = (
            self.reconstruction_weight *reconstruction_loss +
            self.kl_weight * kl_loss + 
            self.perceptual_weight * perceptual_loss
            )
        #
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
            "loss": self.total_loss_tracker.result()
        }
