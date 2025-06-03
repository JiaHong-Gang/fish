import tensorflow as tf
from define_train import VAEModel
from perceptual import compute_perceptual_loss

class Training(VAEModel):
    def train_step(self, data):
        if isinstance(data, tuple):
            x, _= data
        else:
            x = data
        x_img  = x["input_image"]
        x_mask = x["mask"]
        with tf.GradientTape() as tape:

            y_pred, y_maskpred, z_mean, z_log_var = self.vae(x_img, training=True)

            reconstruction_loss, kl_loss, _ = self.vae_loss(x_img, y_pred, z_mean, z_log_var)
            mask_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_mask, y_maskpred))
            perceptual_loss = compute_perceptual_loss(x_img, y_pred)

            total_loss = reconstruction_loss + kl_loss + perceptual_loss + mask_loss
        grads = tape.gradient(total_loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_variables))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.mask_loss_tracker.update_state(mask_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
            "mask_loss": self.mask_loss_tracker.result(),
            "loss": self.total_loss_tracker.result()
        }
    def test_step(self, data):
        #
        if isinstance(data, tuple):
            x, _= data
        else:
            x = data
        x_img = x["input_image"]
        x_mask = x["mask"]
        y_pred, y_maskpred, z_mean, z_log_var = self.vae(x_img, training=False)
        reconstruction_loss, kl_loss, _ = self.vae_loss(x_img, y_pred, z_mean, z_log_var)
        perceptual_loss = compute_perceptual_loss(x_img, y_pred)
        mask_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_mask, y_maskpred))
        total_loss = reconstruction_loss + kl_loss + perceptual_loss + mask_loss
        #
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.mask_loss_tracker.update_state(mask_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
            "mask_loss": self.mask_loss_tracker.result(),
            "loss": self.total_loss_tracker.result()
        }