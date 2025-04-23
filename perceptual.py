import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
#load resnet50 model
resnet50 = ResNet50(include_top = False, weights = "imagenet", input_shape = (512,512,3))
resnet50.trainable = False
perceptual_model = Model(inputs = resnet50.input, outputs = resnet50.get_layer("conv3_block4_out").output) #get model's output of layer"conv3_block4_out"
#define perceptual loss
def compute_perceptual_loss (y_true, y_pred):
    y_true_proc = preprocess_input(y_true * 255.0)
    y_pred_proc = preprocess_input(y_pred * 255.0)
    f_true = perceptual_model(y_true_proc)
    f_pred = perceptual_model(y_pred_proc)
    loss = tf.reduce_mean(tf.square(f_true - f_pred))
    return loss