from loadimage import load_image_and_mask
from imageprocess import process_image

image, mask = load_image_and_mask()
images = process_image(image, is_mask = False)
masks = process_image(mask, is_mask = True)
# 打印前 10 个图像
print("First 10 processed images:")
for i, img in enumerate(images[:10]):
    print(f"Image {i + 1}:")
    print(img)

# 打印前 10 个掩膜
print("\nFirst 10 processed masks:")
for i, mask in enumerate(masks[:10]):
    print(f"Mask {i + 1}:")
    print(mask)

