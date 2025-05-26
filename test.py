
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
import numpy as np


from load_image import load_images
from process_image import process_image
"""
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
    print("original_images mean/std:", original_images.mean(), original_images.std())
    print(">>> output min/max:", recon_imgs.min(),      recon_imgs.max())
    recon_uint8_ceil = np.ceil(recon_imgs * 255.0).clip(0, 255).astype("uint8")

    #    B) （可选）float 保存，让 matplotlib 自行映射
    #       下面在 save 时直接传 recon_imgs[i] 并指定 vmin/vmax

    # 5) 保存并打印
    os.makedirs(save_path, exist_ok=True)
    for i in range(num_samples):
        # 保存原图
        plt.imsave(os.path.join(save_path, f"original_{i}.png"), original_images[i], vmin=0, vmax=1)

        # 保存重构
        plt.imsave(os.path.join(save_path, f"recon_float_{i}.png"), recon_imgs[i], vmin=0, vmax=1)

        # 保存拉伸后的重构
        img = recon_imgs[i]
        img_stretch = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imsave(os.path.join(save_path, f"recon_stretch_{i}.png"), img_stretch, vmin=0, vmax=1)

        print(f"[{i}] recon float min/max = {img.min():.4f}/{img.max():.4f}, after stretch: {img_stretch.min():.4f}/{img_stretch.max():.4f}")
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
"""

MODEL_PATH  = "/home/gang/programs/fish/result/model.h5"
FOLDER_ORIG = "/home/gang/fish/IDdata"
SAVE_DIR    = "/home/gang/programs/fish/test"
NUM_SAMPLES = 5

def load_and_preprocess_images(folder_path, target_size):
    images = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(('.jpg','.png','.jpeg')):
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(folder_path, fname),
                target_size=target_size
            )
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(arr)
    return np.array(images)

if __name__ == "__main__":
    # 1. 加载完整 VAE 并读出它的输入尺寸
    vae = load_model(MODEL_PATH, compile=False)
    # vae.input.shape: (None, H, W, 3)
    _, H, W, _ = vae.input.shape.as_list()
    print(f"✅ Loaded VAE; input size = ({H}, {W})")

    # 2. 裁剪／缩放并归一化原始图像
    os.makedirs(SAVE_DIR, exist_ok=True)
    x_orig = load_and_preprocess_images(FOLDER_ORIG, target_size=(H, W))
    x_orig = x_orig[:NUM_SAMPLES]
    print(f"Loaded {x_orig.shape[0]} images (scaled to {H}×{W})")

    # 3. 拆出 encoder，用 clip_logvar 而不是原始 z_log_var
    encoder = Model(
        inputs=vae.input,
        outputs=[
            vae.get_layer("z_mean").output,
            vae.get_layer("clip_logvar").output
        ]
    )
    z_mean, z_logvar = encoder.predict(x_orig, batch_size=NUM_SAMPLES)
    print(">>> z_mean  min/max/mean/std =",
          f"{z_mean.min():.4f}/{z_mean.max():.4f}/"
          f"{z_mean.mean():.4f}/{z_mean.std():.4f}")
    print(">>> z_logvar min/max/mean/std =",
          f"{z_logvar.min():.4f}/{z_logvar.max():.4f}/"
          f"{z_logvar.mean():.4f}/{z_logvar.std():.4f}")

    # 4. 手动串层：逐层探测 decoder 中间激活
    #    首先重建一个 decoder，从 z_input 到 output
    LATENT_DIM = z_mean.shape[-1]
    z_input = tf.keras.Input(shape=(LATENT_DIM,), name="z_input")
    x = vae.get_layer("decoder_input")(z_input)
    x = vae.get_layer("reshape")(x)
    x = vae.get_layer("transpose_conv4")(x)
    x = vae.get_layer("right_conv4_1")(x)
    x = vae.get_layer("transpose_conv3")(x)
    x = vae.get_layer("right_conv3_1")(x)
    x = vae.get_layer("transpose_conv2")(x)
    x = vae.get_layer("right_conv2_1")(x)
    x = vae.get_layer("transpose_conv1")(x)
    x = vae.get_layer("right_conv1_1")(x)
    x_out = vae.get_layer("output_layer")(x)
    decoder = Model(inputs=z_input, outputs=x_out)

    # 逐层前向并打印统计
    cur_act = tf.convert_to_tensor(z_mean, dtype=tf.float32)
    for layer_name in [
        "decoder_input","reshape",
        "transpose_conv4","right_conv4_1",
        "transpose_conv3","right_conv3_1",
        "transpose_conv2","right_conv2_1",
        "transpose_conv1","right_conv1_1"
    ]:
        layer = decoder.get_layer(layer_name)
        cur_act = layer(cur_act)       # 前向一步
        arr = cur_act.numpy()          # 转 numpy
        print(f"{layer_name:20s} min/max/mean/std = "
              f"{arr.min():8.4f}/{arr.max():8.4f}/"
              f"{arr.mean():8.4f}/{arr.std():8.4f}")

    # 5. 重建测试：A) 均值重建  B) 随机重建
    x_recon_mean = decoder.predict(z_mean)
    eps = np.random.normal(size=z_mean.shape)
    z_rand = z_mean + np.exp(0.5 * z_logvar) * eps
    x_recon_rand = decoder.predict(z_rand)

    # 6. 打印范围 & 保存图片
    print("原图       min/max =", x_orig.min(), x_orig.max())
    print("重建(mean) min/max =", x_recon_mean.min(), x_recon_mean.max())
    print("重建(rand) min/max =", np.nanmin(x_recon_rand), np.nanmax(x_recon_rand))

    for i in range(x_orig.shape[0]):
        plt.imsave(os.path.join(SAVE_DIR, f"orig_{i}.png"),
                   x_orig[i], vmin=0, vmax=1)
        plt.imsave(os.path.join(SAVE_DIR, f"recon_mean_{i}.png"),
                   x_recon_mean[i], vmin=0, vmax=1)
        plt.imsave(os.path.join(SAVE_DIR, f"recon_rand_{i}.png"),
                   x_recon_rand[i], vmin=0, vmax=1)

    print("✅ 重建测试完成，结果保存在", SAVE_DIR)