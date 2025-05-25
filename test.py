
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
import numpy as np


from load_image import load_images
from process_image import process_image

def load_and_preprocess_images(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(('.jpg', '.png', '.jpeg')):
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(720, 512))  # 根据模型输入调整尺寸
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # 归一化到 [0, 1]
            images.append(img)
    return np.array(images)

# 定义计算 MSE 函数
def calculate_mse(original, reconstructed):

    mse = tf.keras.losses.MeanSquaredError()
    return mse(original, reconstructed).numpy()
# 确保图像是 TensorFlow 格式且归一化
def test_model_and_calculate_mse_difference(model, folder_original, save_path, num_samples=5):

    # 创建输出文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载原始和处理后的图像
    original_images = load_and_preprocess_images(folder_original)
   # processed_images = load_and_preprocess_images(folder_processed)

    # 取样数量
    num_samples = min(num_samples, len(original_images))
    print("Original Images Shape:", np.shape(original_images))
    inference_model = Model(inputs=model.input,
                        outputs=model.output[0])

    # 2) 预测
    recon_imgs = inference_model.predict(
        original_images[:num_samples],
        batch_size=num_samples
    )

    # 3) 打印 input/output 范围，检查是否正常
    print(">>> input  min/max:", original_images.min(), original_images.max())
    print(">>> output min/max:", recon_imgs.min(),      recon_imgs.max())
    recon_uint8_ceil = np.ceil(recon_imgs * 255.0).clip(0, 255).astype("uint8")

    #    B) （可选）float 保存，让 matplotlib 自行映射
    #       下面在 save 时直接传 recon_imgs[i] 并指定 vmin/vmax

    # 5) 保存并打印
    os.makedirs(save_path, exist_ok=True)
    for i in range(num_samples):
    # 直接显示 float 图像，让 matplotlib 把 [0,1]→[0,255]
         plt.imsave(os.path.join(save_path, f"recon_float_{i}.png"),
               recon_imgs[i], vmin=0, vmax=1)
         print(f"[{i}] recon float min/max = "
             f"{recon_imgs[i].min():.4f}/{recon_imgs[i].max():.4f}")

 # 路径配置
folder_original = "/home/gang/fish/IDdata"  # 替换为原始图像文件夹路径
#folder_processed = "/home/gou/Programs/fish/test_image/gaussian_image_test"  # 替换为处理后图像文件夹路径
save_path = "/home/gang/programs/fish/test"  # 替换为结果保存路径

# 加载模型
model_path = "/home/gang/programs/fish/result/model.h5"
model = load_model(model_path)
print("✅ 模型已成功加载！")

# 测试模型并计算 MSE 差异
test_model_and_calculate_mse_difference(model, folder_original,save_path)
