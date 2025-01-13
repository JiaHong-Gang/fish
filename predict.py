import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from add_noisy import map_func
from load_image import load_images
from process_image import process_image
import os

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
    """计算均方误差（MSE）"""
    mse = tf.keras.losses.MeanSquaredError()
    return mse(tf.convert_to_tensor(original, dtype=tf.float32),
               tf.convert_to_tensor(reconstructed, dtype=tf.float32)).numpy()

# 创建保存图像的文件夹
def create_output_folder(folder_path):
    """创建用于保存图像的文件夹"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# 保存图像函数
def save_image(image, filename):
    """保存图像到指定路径"""
    plt.imsave(filename, image)

# 测试模型并保存图像
def test_model_and_save_images(model_to_test, test_images, num_samples=5, save_path="/home/gou/Programs/fish/result"):
    """使用加载的模型进行图像重建，并保存图像与计算 MSE"""
    create_output_folder(save_path)  # 创建保存目录

    # 使用 map_func 进行图像噪声添加
    inputs, original_images = map_func(test_images[:num_samples])
    predicted_images = model_to_test.predict(inputs)

    # 计算 MSE
    original_mse = calculate_mse(original_images, predicted_images)
    noisy_mse = calculate_mse(inputs['input_image'], predicted_images)

    print(f"\n✅ 原始图像 MSE: {original_mse:.4f}")
    print(f"✅ 噪音图像 MSE: {noisy_mse:.4f}\n")

    # 可视化与保存图像
    plt.figure(figsize=(15, 7))
    for i in range(num_samples):
        # 原始图像
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(original_images[i])
        plt.axis('off')
        plt.title("Original Image")
        save_image(original_images[i], os.path.join(save_path, f"original_image_{i}.png"))

        # 加噪图像
        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(inputs['input_image'][i].numpy())
        plt.axis('off')
        plt.title("Noisy Image")
        save_image(inputs['input_image'][i].numpy(), os.path.join(save_path, f"noisy_image_{i}.png"))

        # 重建图像
        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(predicted_images[i])
        plt.axis('off')
        plt.title("Reconstructed Image")
        save_image(predicted_images[i], os.path.join(save_path, f"reconstructed_image_{i}.png"))

    plt.tight_layout()
    plt.show()

# 运行测试与保存图像
test_model_and_save_images(model, images)




