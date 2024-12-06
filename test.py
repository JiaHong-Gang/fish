from loadimage import load_image_and_mask
from imageprocess import process_image

image, mask = load_image_and_mask()
images = process_image(image, is_mask = False)
masks = process_image(mask, is_mask = True)

