import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np

# 确保图像是 TensorFlow 格式且归一化
def load_and_preprocess_images(folder_path):
    """加载并归一化图像"""
    images = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(('.jpg', '.png', '.jpeg')):
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(512, 512))  # 根据模型输入调整尺寸
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # 归一化到 [0, 1]
            images.append(img)
    return np.array(images)

# 定义计算 MSE 函数
def calculate_mse(original, reconstructed):
    """计算均方误差（MSE）"""
    mse = tf.keras.losses.MeanSquaredError()
    return mse(original, reconstructed).numpy()

# 测试模型并计算 MSE 差异
def test_model_and_calculate_mse_difference(model, folder_original, folder_processed, save_path, num_samples=5):
    """测试模型并计算原始图像与处理后图像的 MSE 差异"""
    # 创建输出文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载原始和处理后的图像
    original_images = load_and_preprocess_images(folder_original)
    processed_images = load_and_preprocess_images(folder_processed)

    # 取样数量
    num_samples = min(num_samples, len(original_images), len(processed_images))

    # 使用模型预测
    reconstructed_original = model.predict(original_images[:num_samples])
    reconstructed_processed = model.predict(processed_images[:num_samples])

    # 计算 MSE
    mse_original = calculate_mse(original_images[:num_samples], reconstructed_original)
    mse_processed = calculate_mse(processed_images[:num_samples], reconstructed_processed)

    print(f"✅ 原始图像 MSE: {mse_original:.4f}")
    print(f"✅ 处理后图像 MSE: {mse_processed:.4f}")

    # 保存图像
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        # 原始图像
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(original_images[i])
        plt.axis('off')
        plt.title("Original Image")
        plt.imsave(os.path.join(save_path, f"original_image_{i}.png"), original_images[i])

        # 处理后图像
        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(processed_images[i])
        plt.axis('off')
        plt.title("Processed Image")
        plt.imsave(os.path.join(save_path, f"processed_image_{i}.png"), processed_images[i])

        # 重建图像
        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(reconstructed_processed[i])
        plt.axis('off')
        plt.title("Reconstructed Processed")
        plt.imsave(os.path.join(save_path, f"reconstructed_image_{i}.png"), reconstructed_processed[i])

    plt.tight_layout()
    plt.show()

# 路径配置
folder_original = "/home/gou/Programs/fish/test/OriginalImages"  # 替换为原始图像文件夹路径
folder_processed = "/home/gou/Programs/fish/test/ProcessedImages"  # 替换为处理后图像文件夹路径
save_path = "/home/gou/Programs/fish/test"  # 替换为结果保存路径

# 加载模型
model_path = "/home/gou/Programs/fish/result/model.h5"
model = load_model(model_path)
print("✅ 模型已成功加载！")

# 测试模型并计算 MSE 差异
test_model_and_calculate_mse_difference(model, folder_original, folder_processed, save_path)