import numpy as np
import os
import matplotlib.pyplot as plt
from config import batch_size

def overlay_segmentation(image, prediction, alpha=0.5, color=(1, 0, 0)):
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = np.stack([image] * 3, axis=-1)

    prediction = np.squeeze(prediction)  # 确保 prediction 是 2D 的

    overlay = np.zeros_like(image, dtype=np.float32)
    overlay[:, :, 0] = prediction * color[0]
    overlay[:, :, 1] = prediction * color[1]
    overlay[:, :, 2] = prediction * color[2]

    # 叠加图像
    blended = image.astype(np.float32) / 255.0
    blended = blended * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


def visualize_overlays_and_save(images, masks, predictions, alpha=0.5, output_dir="/home/gou/Programs/fish/result/segment_image"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_images = min(5, len(images))
    for i in range(num_images):
        plt.figure(figsize=(15, 5))

        # 原图像
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(images[i], cmap='gray' if images[i].ndim == 2 else None)
        plt.axis("off")

        # 原图像 + 预测结果
        overlay = overlay_segmentation(images[i], predictions[i], alpha=alpha)
        plt.subplot(1, 3, 2)
        plt.title("Overlay (Prediction)")
        plt.imshow(overlay)
        plt.axis("off")

        # 真实标签
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(masks[i], cmap='gray')
        plt.axis("off")

        # 保存图片
        plt.savefig(os.path.join(output_dir, f"image_{i}.png"))
        plt.close()

def prediction(x_val, y_val, model):
    model.load_weights("/home/gou/Programs/fish/result/model_weight.h5")
    predictions = model.predict(x_val, batch_size=batch_size)
    binary_predictions = (predictions > 0.5).astype(np.uint8)
    visualize_overlays_and_save(x_val, y_val, binary_predictions, alpha=0.5)
    print(f"segment image has been saved")
