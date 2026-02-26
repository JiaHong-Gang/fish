
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np

def load_and_preprocess_images(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(('.jpg', '.png', ".JPG")):
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(1088, 768))  # resize model input size
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # normalized image to [0, 1]
            images.append(img)
    return np.array(images)

# define MSE function
def calculate_mse(original, reconstructed):

    mse = tf.keras.losses.MeanSquaredError()
    return mse(original, reconstructed).numpy()
def test_model_and_calculate_mse_difference(model, folder_original, save_path, num_samples=40):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    original_images = load_and_preprocess_images(folder_original)

    num_samples = min(num_samples, len(original_images))
    print(f"samples are: {num_samples}")
    print("Original Images Shape:", np.shape(original_images))

    reconstructed_original_list = []
    reconstructed_mask_list = []

    for i in range(num_samples):
        single_image = np.expand_dims(original_images[i], axis=0)

        output = model.predict(single_image)

        if isinstance(output, list):
            reconstructed_original_list.append(output[0][0])
            reconstructed_mask_list.append(output[1][0])
        else:
            reconstructed_original_list.append(output[0])

        print(f"🔄 Processing {i+1}/{num_samples}", end='\r')

    reconstructed_original = np.array(reconstructed_original_list)
    reconstructed_mask = np.array(reconstructed_mask_list) if reconstructed_mask_list else None

    print("\nReconstructed Original Shape:", np.shape(reconstructed_original))
    print("Reconstructed Mask Shape:", np.shape(reconstructed_mask))

    mse_original = calculate_mse(original_images[:num_samples], reconstructed_original)
    print(f"✅ original MSE: {mse_original:.8f}")

    plt.figure(figsize=(15, 10))
    for i in range(num_samples):

        plt.subplot(3, num_samples, i + 1)
        plt.imshow(original_images[i])
        plt.axis('off')
        plt.title("Original Image")
        plt.imsave(os.path.join(save_path, f"original_image_{i}.png"), original_images[i])

        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(reconstructed_original[i])
        plt.axis('off')
        plt.title("Reconstructed Output")
        plt.imsave(os.path.join(save_path, f"reconstructed_image_{i}.png"), reconstructed_original[i])

        """
        #make images
        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        mask = reconstructed_mask[i]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask)
        plt.imshow(mask, cmap = "gray")
        plt.axis("off")
        plt.title("Reconstructed mask")
        plt.imsave(os.path.join(save_path,f"reconstructed_mask_{i}.png"), mask, cmap = "gray")
        
    plt.tight_layout()
    plt.show()
        """
# set path
folder_original = "/home/gang/fish/IDdata"
save_path = "/home/gang/programs/fish/test"

# load model
model_path = "/home/gang/programs/fish/result/vae_model_upsample"
vae_model = keras.models.load_model(model_path)
print("✅ model has been loaded！")

# calculate mse
test_model_and_calculate_mse_difference(vae_model, folder_original, save_path)
