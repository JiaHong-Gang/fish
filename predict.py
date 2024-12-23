import numpy as np
import os
import cv2
from skimage.transform import resize
import umap.umap_ as umap
from tensorflow.keras.models import Model
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
def split_image(x_val,y_val , block_size = 256, target_size = 512):
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
                block_img = np.array(block_img, dtype = np.float32)
                block_mask = np.array(block_mask, dtype = np.float32)
                block_img_resized = cv2.resize(block_img,(target_size, target_size), interpolation=cv2.INTER_LINEAR)
                block_mask_resized = cv2.resize(block_mask,(target_size, target_size), interpolation=cv2.INTER_NEAREST)
                block_images.append(block_img_resized)
                block_masks.append(block_mask_resized)
        print(f"image processed into {len(block_images)} blocks")
        all_block_images.append(block_images)
        all_block_masks.append(block_masks)
    return all_block_images, all_block_masks

def calculate_iou(pred, mask):
    pred_binary = (pred > 0.5).astype(np.int32)
    mask_binary = mask.astype(np.int32)
    intersection = np.logical_and(pred_binary, mask_binary).sum()
    union = np.logical_or(pred_binary, mask_binary).sum()
    return intersection / union if union > 0 else 0

def visualize_result(block_images, predictions, save_dir="/home/gou/Programs/fish/result/segment_block_image"):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Total blocks to save: {len(block_images)}")  

    for idx, (img, pred) in enumerate(zip(block_images, predictions)):
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Original Block {idx + 1}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap="gray")
        plt.title(f"Predicted Block {idx + 1}")
        plt.axis("off")

        save_path = os.path.join(save_dir, f"block_{idx + 1}.jpg")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    print(f"All blocks saved to {save_dir}")


def predict_block_image(x_val, y_val, model, block_size = 256):
    all_block_images, all_block_masks = split_image(x_val, y_val, block_size)
    iou_list = []
    all_images = []
    all_predictions = []
    model.load_weights("/home/gou/Programs/fish/result/model_weight.h5")
    for batch_images, batch_masks in zip(all_block_images, all_block_masks):
        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        batch_predictions = model.predict(batch_images, batch_size=batch_size)

        for idx, (img, pred, mask) in enumerate(zip(batch_images, batch_predictions, batch_masks)):
            if pred.ndim == 3:
                pred = pred[:,:,0]
            pred_resized = cv2.resize(pred, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask,(block_size, block_size), interpolation=cv2.INTER_NEAREST)
            img_resized = cv2.resize(img, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
            pred_resized = (pred_resized > 0.5).astype(np.int32)
            mask_resized = mask_resized.astype(np.int32)
            iou = calculate_iou(pred_resized, mask_resized)
            iou_list.append(iou)
            print(f"block {idx +1} , iou = {iou:.2f}")
            all_images.append(img_resized)
            all_predictions.append(pred_resized)
    mean_iou = np.mean(iou_list)
    print(f"mean iou is {mean_iou:.2f}")
    visualize_result(all_images, all_predictions)
#------------------------UMAP-------------------------
def feature_dim_reduce(x_val, y_val, model):
    model.load_weights("/home/gou/Programs/fish/result/model_weight.h5")
    print(f"model weight has been loaded")
    save_dir = "/home/gou/Programs/fish/result/"
    os.makedirs(save_dir, exist_ok=True)
    layer_name = "bottom_Conv2"
    features = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)
    num_samples = x_val.shape[0]
    all_features = []

    for i in range(0, num_samples,batch_size):
        batch_images = x_val[i: i+batch_size]
        feature = features.predict(batch_images)
        all_features.append(feature)
    all_features = np.concatenate(all_features, axis = 0)
    print(f"Feature extracted , shape: ", all_features.shape)
    target_height, target_width = all_features.shape[1:3]
    y_val_resized = np.array([resize(mask, (target_height, target_width), order=0, preserve_range=True, anti_aliasing=False)
                              for mask in y_val])
    print(f"Resized masks shape: {y_val_resized.shape}")
    if len(y_val_resized.shape) == 4 and y_val_resized.shape[-1] == 1:
        y_val_resized = y_val_resized.squeeze(axis=-1)
    print(f"Adjusted Resized masks shape: {y_val_resized.shape}")
    flattened_features = all_features.reshape(-1, all_features.shape[-1])
    flattened_masks = y_val.flatten()
    print(f"Flattened features shape: ", flattened_features.shape)
    print(f"Flattened masks shape: ", flattened_masks.shape)

    reducer = umap.UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.1, random_state = 42)
    embedding = reducer.fit_transform(flattened_features)
    print(f"UMAP embedding shape :", embedding.shape)

    plt.figure(figsize = (10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c="gray", s=1, alpha=0.3, label="Intra-Class Distribution")
    plt.scatter(embedding[:, 0], embedding[:, 1], c=flattened_masks, cmap='coolwarm', s=1, alpha=0.7,
                label="Inter-Class Distribution")
    plt.colorbar(label="Class (Foreground=1, Background=0)")
    plt.title("Combined Distribution: Inter-Class and Intra-Class")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    save_path = os.path.join(save_dir, "umap.jpg")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


