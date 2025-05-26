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
        if file_path.endswith(('.jpg', '.png', '.jpeg')):
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(720, 512))  # resize model input size
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # normalized image to [0, 1]
            images.append(img)
    return np.array(images)

# define MSE function
def calculate_mse(original, reconstructed):

    mse = tf.keras.losses.MeanSquaredError()
    return mse(original, reconstructed).numpy()

def test_model_and_calculate_mse_difference(model, folder_original, save_path, num_samples=5):

    # make output file
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load original images and processed images
    original_images = load_and_preprocess_images(folder_original)
    #processed_images = load_and_preprocess_images(folder_processed)

    # sample number
    num_samples = min(num_samples, len(original_images))
    print("Original Images Shape:", np.shape(original_images))
    #print("Processed Images Shape:", np.shape(processed_images))

    # use model tp predict ,only use first output
    output_original = model.predict(original_images[:num_samples])
    #output_processed = model.predict(processed_images[:num_samples])

    # if model back multiply output ,only use first
    reconstructed_original = output_original[0] if isinstance(output_original, list) else output_original
    #reconstructed_processed = output_processed[0] if isinstance(output_processed, list) else output_processed

    print("Reconstructed Original Shape:", np.shape(reconstructed_original))
    #print("Reconstructed Processed Shape:", np.shape(reconstructed_processed))

    # calculate mse
    mse_original = calculate_mse(original_images[:num_samples], reconstructed_original)
    #mse_processed = calculate_mse(processed_images[:num_samples], reconstructed_processed)

    print(f"✅ original MSE: {mse_original:.8f}")
    #print(f"✅ processed MSE: {mse_processed:.8f}")

    # save image
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        # original images
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(original_images[i])
        plt.axis('off')
        plt.title("Original Image")
        plt.imsave(os.path.join(save_path, f"original_image_{i}.png"), original_images[i])

        # reconstructed images
        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(reconstructed_original[i])
        plt.axis('off')
        plt.title("Reconstructed Processed")
        plt.imsave(os.path.join(save_path, f"reconstructed_image_{i}.png"), reconstructed_original[i])

    plt.tight_layout()
    plt.show()

# set path
folder_original = "/home/gang/fish/IDdata"
save_path = "/home/gang/programs/fish/test"

# load model
model_path = "/home/gang/programs/fish/result/vae_model"
vae_model = keras.models.load_model(model_path)
print("✅ model has been loaded！")

# calculate mse
test_model_and_calculate_mse_difference(vae_model, folder_original, save_path)

















