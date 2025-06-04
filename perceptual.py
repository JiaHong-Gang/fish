import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
#load resnet50 model
layer_name = ["conv1_relu","conv2_block3_out","conv3_block4_out"]
#layer_name = ["conv2_block3_out","conv3_block4_out", "conv4_block5_out"]
#layer_name = ["conv3_block4_out"]
resnet50 = ResNet50(include_top = False, weights = "imagenet", input_shape = (1088,768,3))
resnet50.trainable = False
outputs = []
for name in layer_name:
    output = resnet50.get_layer(name).output
    outputs.append(output)
perceptual_model = Model(inputs = resnet50.input, outputs = outputs) #get model's output of layer"conv3_block4_out"
#define perceptual loss
def compute_perceptual_loss (y_true, y_pred):
    loss = 0.0
    y_true_proc = preprocess_input(y_true * 255.0)
    y_pred_proc = preprocess_input(y_pred * 255.0)
    f_true = perceptual_model(y_true_proc)
    f_pred = perceptual_model(y_pred_proc)
    if not isinstance(f_true, (list, tuple)):
        f_true = [f_true]
        f_pred = [f_pred]
    weights = [3.5, 0.5, 0.5]
    #weights = [1.0]*len(f_true)
    for ft, fp, w in zip(f_true, f_pred, weights):
        loss += w * tf.reduce_mean(tf.square(ft - fp))
    return loss
