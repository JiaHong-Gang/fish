from config import batch_size, epochs, ht_img, wd_img
from loadimage import load_image_and_mask
from metric import iou_metric
from unet_model import unet
#from metric import iou_metric
from imageprocess import process_image
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
def train_model():
    images, mask = load_image_and_mask()
    images = process_image(images, is_mask = False)
    masks = process_image(mask, is_mask= True)
    x_train, x_val, y_train, y_val = train_test_split(images, masks ,test_size = 0.2, random_state = 42)
    model = unet(input_shape=(ht_img, wd_img, 3))
    # compile model
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy", iou_metric])
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size, epochs=epochs,
        verbose=1, validation_data = (x_val, y_val), shuffle=True
    )
    model.save_weights("/home/gou/Programs/fish/result/model_weight.h5")
    return model, history, x_val, y_val