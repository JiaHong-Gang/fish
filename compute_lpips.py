import cv2
import os
import numpy as np
import torch
import lpips
import pyiqa
folder_path = "/Users/gangjiahong/Desktop/result/test/tivp_c1ruv2b3c3b4_mask/UpSample"
def load_image(folder_path):
    original_img = []
    reconstructed_img = []
    original_name = []
    reconstructed_name = []
    MAX_INDEX = 4
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".png"):
            continue
        if filename.lower().startswith(("original_mask_","reconstructed_mask_")):
            continue
        if filename.startswith("original_image_"):
            idx = int(filename[len("original_image_"):-len(".png")])
            if idx > MAX_INDEX:
                continue
        elif filename.startswith("reconstructed_image_"):
            idx = int(filename[len("reconstructed_image_"):-len(".png")])
            if idx > MAX_INDEX:
                continue
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if "original_image_" in filename:
            original_img.append(img)
            original_name.append(filename)
        elif "reconstructed_image_" in filename:
            reconstructed_img.append(img)
            reconstructed_name.append(filename)
    return original_img, reconstructed_img,original_name, reconstructed_name
def to_tensor(img):
    image = img.astype(np.float32) / 255.0
    image = image *2.0 -1.0
    image_t = torch.from_numpy(image).permute(2, 0, 1)
    image_t = image_t.unsqueeze(0)
    return image_t
original_image, reconstructed_image , on, rn= load_image(folder_path)
print(f"number of original images are {len(original_image)} \nnumber of reconstructed images are {len(reconstructed_image)}")
for i in range(len(original_image)):
    print(f"original name:{on[i]}, reconstructed name: {rn[i]}")
loss_fn_vgg = lpips.LPIPS(net = "vgg")
niqe_metric = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)
lpips_score = []
niqe_score = []
for i in range(len(original_image)):
    img0 = to_tensor(original_image[i])
    img1 = to_tensor(reconstructed_image[i])
    with torch.no_grad():
        dist = loss_fn_vgg(img0, img1)
    lpips_score.append(dist.item())
    img1_norm = (img1 + 1) / 2.0
    with torch.no_grad():
        niqe_val = niqe_metric(img1_norm)
    niqe_score.append(niqe_val.item())
print("LPIPS Scores (Lower is better):")
for s in lpips_score:
    print(f"{s:.4f}")
print("NIQE Scores (Lower is better):")
for n in niqe_score:
    print(f"{n:.4f}")
