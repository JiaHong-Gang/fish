import matplotlib.pyplot as plt
import os
import numpy as np

def predictions(model, images, masks, save_path = "/home/gang/programs/fish/test", num=3):
    os.makedirs(save_path, exist_ok=True)  
    preds = model.predict(images[:num])

    for i in range(num):
        plt.figure(figsize=(15, 5))

        # input image
        input_img = images[i]
        plt.subplot(2, 4, 1)
        plt.imshow(input_img)
        plt.title("Input Image")
        plt.axis("off")
        plt.imsave(os.path.join(save_path, f"input_image_{i}.png"), (input_img * 255).astype(np.uint8))

        # Ground Truth masks
        for j, label in enumerate(["Body", "Red", "White"]):
            gt_mask = masks[i][:, :, j]
            plt.subplot(2, 4, j + 2)
            plt.imshow(gt_mask, cmap="gray")
            plt.title(f"GT - {label}")
            plt.axis("off")
            plt.imsave(os.path.join(save_path, f"gt_{label.lower()}_{i}.png"), gt_mask, cmap="gray")

        # predict mask
        for j, label in enumerate(["Body", "Red", "White"]):
            pred_mask = (preds[i][:, :, j] > 0.5).astype(np.float32)
            plt.subplot(2, 4, j + 5)
            plt.imshow(pred_mask, cmap="gray")
            plt.title(f"Pred - {label}")
            plt.axis("off")
            plt.imsave(os.path.join(save_path, f"pred_{label.lower()}_{i}.png"), pred_mask, cmap="gray")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"summary_{i}.png"))  
        plt.close()