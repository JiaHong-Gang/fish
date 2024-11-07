from config import batch_size, epochs, ht_img, wd_img
from resnet_model import ResNet_18
from imageprocess import process_image, label
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
def train_model():
    images = process_image()
    labels = label(images)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 42)
    model = ResNet_18(input_shape=(ht_img, wd_img, 3))
    # compile model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.000001), metrics=["accuracy"])
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size, epochs=epochs,
        verbose=1, validation_data = (x_test, y_test), shuffle=True
    )
    model.save_weights("/home/gou/Programs/fish/result/model_weight.h5")
    return model, history, x_test, y_test