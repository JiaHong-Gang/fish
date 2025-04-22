
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.model_selection import train_test_split
from load_image import load_images
from process_image import process_image

def load_and_preprocess_images(folder_path):
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

    mse = tf.keras.losses.MeanSquaredError()
    return mse(original, reconstructed).numpy()
# 确保图像是 TensorFlow 格式且归一化
def test_model_and_calculate_mse_difference(model, folder_original, folder_processed, save_path, num_samples=5):

    # 创建输出文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载原始和处理后的图像
    original_images = load_and_preprocess_images(folder_original)
    processed_images = load_and_preprocess_images(folder_processed)

    # 取样数量
    num_samples = min(num_samples, len(original_images), len(processed_images))
    print("Original Images Shape:", np.shape(original_images))
    print("Processed Images Shape:", np.shape(processed_images))

    # 使用模型预测，只取第一个输出
    output_original = model.predict(original_images[:num_samples])
    output_processed = model.predict(processed_images[:num_samples])

    # 如果模型返回多个输出，只取第一个输出
    reconstructed_original = output_original[0] if isinstance(output_original, list) else output_original
    reconstructed_processed = output_processed[0] if isinstance(output_processed, list) else output_processed

    print("Reconstructed Original Shape:", np.shape(reconstructed_original))
    print("Reconstructed Processed Shape:", np.shape(reconstructed_processed))

    # 计算 MSE
    mse_original = calculate_mse(original_images[:num_samples], reconstructed_original)
    mse_processed = calculate_mse(processed_images[:num_samples], reconstructed_processed)

    print(f"✅ 原始图像 MSE: {mse_original:.8f}")
    print(f"✅ 处理后图像 MSE: {mse_processed:.8f}")

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
folder_original = "/home/gang/Programs/fish/test_image/gaussian_image_test"  # 替换为原始图像文件夹路径
folder_processed = "/home/gang/Programs/fish/test_image/gaussian_image_test"  # 替换为处理后图像文件夹路径
save_path = "/home/gang/Programs/fish/test"  # 替换为结果保存路径

# 加载模型
model_path = "/home/gou/Programs/fish/result/model.h5"
model = load_model(model_path)
print("✅ 模型已成功加载！")

# 测试模型并计算 MSE 差异
test_model_and_calculate_mse_difference(model, folder_original, folder_processed, save_path)
"""
# 1️⃣ 加载模型
model_path = "/home/gou/Programs/fish/result/model.h5"
model = load_model(model_path, compile=False)  # 不编译，因为我们手动计算 loss
print("✅ Model has been loaded!")

# 2️⃣ 预处理数据（确保数据类型是 float32）
images = load_images()  # 加载图像
images = process_image(images)  # 预处理图像

x_train, x_val = train_test_split(images, test_size=0.2, random_state=42)
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)

# 3️⃣ 处理训练集 & 测试集
batch_size = 2  # 你训练时的 batch_size
train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .map(lambda x: tf.cast(x, tf.float32))
    .batch(batch_size)
)
val_dataset = (
    tf.data.Dataset.from_tensor_slices(x_val)
    .map(lambda x: tf.cast(x, tf.float32))
    .batch(batch_size)
)

# 4️⃣ 计算损失
def compute_losses(dataset, dataset_type="Train"):
    total_loss = tf.Variable(0.0, dtype=tf.float32)
    total_reco_loss = tf.Variable(0.0, dtype=tf.float32)
    total_kl_loss = tf.Variable(0.0, dtype=tf.float32)
    num_batches = tf.Variable(0, dtype=tf.int32)

    for x_batch in dataset:
        # 预测输出
        y_pred, z_mean, z_log_var = model(x_batch, training=False)

        # 计算 MSE 重建损失
        mse_loss = tf.reduce_mean(tf.square(x_batch - y_pred))

        # 计算 KL Loss（TensorFlow 计算，防止溢出）
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        # 计算总损失
        loss = mse_loss + kl_loss

        # 更新累加损失
        total_loss.assign_add(loss)
        total_reco_loss.assign_add(mse_loss)
        total_kl_loss.assign_add(kl_loss)
        num_batches.assign_add(1)

    # 计算平均损失
    avg_loss = total_loss / tf.cast(num_batches, tf.float32)
    avg_reco_loss = total_reco_loss / tf.cast(num_batches, tf.float32)
    avg_kl_loss = total_kl_loss / tf.cast(num_batches, tf.float32)

    print(f"\n📊 {dataset_type} - Total Loss: {avg_loss:.8f}, MSE Loss: {avg_reco_loss:.8f}, KL Loss: {avg_kl_loss:.8f}")
    
    return avg_loss.numpy(), avg_reco_loss.numpy(), avg_kl_loss.numpy()

# 计算 **最后一个 epoch** 的损失
final_train_loss, final_train_reco_loss, final_train_kl_loss = compute_losses(train_dataset, dataset_type="Train")
final_val_loss, final_val_reco_loss, final_val_kl_loss = compute_losses(val_dataset, dataset_type="Validation")
"""