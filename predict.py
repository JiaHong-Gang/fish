import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from add_noisy import map_func
from load_image import load_images
from process_image import process_image
import os
import numpy as np

# 加载并预处理图像
images = load_images()
images = process_image(images)

# 确保图像是 TensorFlow 格式且归一化
if images.max() > 1.0:
    images = images / 255.0

# 加载模型
model_path = "/home/gou/Programs/fish/result/model.h5"
model = load_model(model_path)
print("✅ 模型已成功加载！")

# 定义计算 MSE 函数
def calculate_mse(original, reconstructed):
    """计算均方误差（MSE），确保输入为规则的 Tensor"""
    # 确保输入是 NumPy 数组或 TensorFlow 张量
    if isinstance(original, list):
        original = tf.convert_to_tensor(np.array(original), dtype=tf.float32)
    if isinstance(reconstructed, list):
        reconstructed = tf.convert_to_tensor(np.array(reconstructed), dtype=tf.float32)
        
    mse = tf.keras.losses.MeanSquaredError()
    return mse(original, reconstructed).numpy()

# 创建保存图像的文件夹
def create_output_folder(folder_path):
    """创建用于保存图像的文件夹"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_image(image, filename):
    """保存图像，并确保像素值在 [0,1] 范围内"""
    # 将 TensorFlow 张量转换为 NumPy 数组
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # 将像素值限制在 [0, 1] 范围内
    image = np.clip(image, 0.0, 1.0)

    # 保存图像
    plt.imsave(filename, image)


# 测试模型并保存图像
def test_model_and_save_images(model_to_test, test_images, num_samples=5, save_path="/home/gou/Programs/fish/result/test_results"):
    """使用加载的模型进行图像重建，并保存图像与计算 MSE"""
    create_output_folder(save_path)  # 创建保存目录

    # 使用 map_func 进行图像噪声添加
    inputs, original_images = map_func(test_images[:num_samples])
    predicted_images, _, _ = model_to_test.predict(inputs)

    # 计算 MSE
    original_mse = calculate_mse(original_images, predicted_images)
    noisy_mse = calculate_mse(inputs['input_image'], predicted_images)

    print(f"\n✅ 原始图像 MSE: {original_mse:.4f}")
    print(f"✅ 噪音图像 MSE: {noisy_mse:.4f}")

    # 可视化与保存图像
    plt.figure(figsize=(15, 7))
    for i in range(num_samples):
        # 原始图像
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(np.clip(original_images[i], 0, 1))
        plt.axis('off')
        plt.title("Original Image")
        save_image(original_images[i], os.path.join(save_path, f"original_image_{i}.png"))

        # 加噪图像
        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(np.clip(inputs['input_image'][i].numpy(), 0, 1))
        plt.axis('off')
        plt.title("Noisy Image")
        save_image(inputs['input_image'][i].numpy(), os.path.join(save_path, f"noisy_image_{i}.png"))

        # 重建图像
        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(np.clip(predicted_images[i], 0, 1))
        plt.axis('off')
        plt.title("Reconstructed Image")
        save_image(predicted_images[i], os.path.join(save_path, f"reconstructed_image_{i}.jpeg"))
        print("predict images has been saved")
    plt.tight_layout()
    plt.show()


# 运行测试与保存图像
test_model_and_save_images(model, images)




