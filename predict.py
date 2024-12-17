import numpy as np
import os
import matplotlib.pyplot as plt
from config import batch_size

#-------------------show segmentation result-------------------------------

def overlay_segmentation(image, prediction, alpha=0.5, color=(1, 0, 0)):
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = np.stack([image] * 3, axis=-1)

    prediction = np.squeeze(prediction)  #make sure images are 2D

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
        plt.savefig(os.path.join(output_dir, f"image_{i}.jpg"))
        plt.close()

def prediction(x_val, y_val, model):
    model.load_weights("/home/gou/Programs/fish/result/model_weight.h5")
    predictions = model.predict(x_val, batch_size=batch_size)
    binary_predictions = (predictions > 0.5).astype(np.uint8)
    visualize_overlays_and_save(x_val, y_val, binary_predictions, alpha=0.5)
    print(f"segment image has been saved")

#-------------------show block segmentation result--------------------
def split_image(x_val,y_val , block_size = 256):
    all_block_images = []
    all_block_masks = []
    for img, mask in zip(x_val[:5], y_val[:5]):
        block_images = []
        block_masks = []
        h, w = img.shape[:2]
        for i in range(0,h,block_size):
            for j in range(0,w,block_size):
                block_img = img[i:i + block_size, j:j + block_size]
                block_mask = mask[i:i + block_size ,j:j + block_size]
                block_images.append(block_img)
                block_masks.append(block_mask)
        all_block_images.append(block_images)
        all_block_masks.append(block_masks)
    return all_block_images, all_block_masks

def calculate_iou(pred, mask):
    pred_binary = (pred > 0.5).astype(np.int32)
    mask_binary = mask.astype(np.int32)
    intersection = np.logical_and(pred_binary, mask_binary).sum()
    union = np.logical_or(pred_binary, mask_binary).sum()
    return intersection / union if union > 0 else 0

def visualize_result(block_images, predictions, save_dir = "/home/gou/Programs/fish/result/segment_block_image"):
    os.makedirs(save_dir, exist_ok = True)
    plt.figure(figsize = (12,12))

    num_to_show = min(4,len(block_images),len(predictions))

    for i in range(num_to_show):
        plt.subplot(4, 3, i*3 +1)
        plt.imshow(block_images[i])
        plt.title("Original image block")
        plt.axis("off")

        plt.subplot(4, 3, i*3 +2)
        plt.imshow(predictions[i], cmap = "gray")
        plt.title("predicted image block")
        plt.axis("off")
        save_path = os.path.join(save_dir, f"image {i}.jpg")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def predict_block_image(x_val, y_val, model):
    all_block_images, all_block_masks = split_image(x_val, y_val, 256)
    iou_list = []
    all_images = []
    all_predictions = []
    model.load_weights("/home/gou/Programs/fish/result/model_weight.h5")
    for batch_images, batch_masks in zip(all_block_images, all_block_masks):
        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        batch_predictions = model.predict(batch_images, batch_size=batch_size)

        for pred, mask in zip(batch_predictions, batch_masks):
            if pred.ndim == 3:
                pred = pred[:,:,0]
            iou = calculate_iou(pred, mask)
            iou_list.append(iou)
            print(f"iou is {iou:.2f}")
        all_images.append(batch_images)
        all_predictions.append(batch_predictions)
    mean_iou = np.mean(iou_list)
    print(f"mean iou is {mean_iou:.2f}")
    visualize_result(all_images, all_predictions)
