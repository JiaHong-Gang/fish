import cv2
import os

def load_image_and_mask(identifier = "ID"):
    img_folder = "/home/gou/fish/IDdata"
    mask_folder = "/home/gou/fish/masks"
    # read image ID
    img_dict = {}
    for img_file in os.listdir(img_folder):
        if img_file.startswith('.'): #ignore . file
            continue
        img_id = [part for part in img_file.split("_") if identifier in part] #split file name
        if img_id:
            img_dict[img_id[0]] = os.path.join(img_folder, img_file) # add image path
    #read mask ID
    mask_dict = {}
    for mask_file in os.listdir(mask_folder):
        if mask_file.startswith('.'): #ignore . file
            continue
        mask_id = [part for part in mask_file.split("_") if identifier in part] #split file name
        if mask_id:
            mask_dict[mask_id[0]] = os.path.join(mask_folder, mask_file)  # add mask path
    #compair image and mask ID
    common_ids = set(img_dict.keys()) & set(mask_dict.keys())
    if not common_ids:
        raise ValueError("No matching images and masks found!")
    images = []
    masks = []
    for common_id in common_ids:
        img_path = img_dict[common_id]
        mask_path = mask_dict[common_id]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # read as RGB image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # read as gray image

        images.append(img)
        masks.append(mask)

    print(f"Loaded {len(images)} images and {len(masks)} masks.")
    return images, masks


