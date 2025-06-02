import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

#load unet model
model_path = "/home/gang/programs/fish/unet_model_weight/model.h5"
unet_model = load_model(model_path, compile = False)
layer_name = ["left_Conv1_2","left_Conv2_2","left_Conv3_2"]
unet_model.trainable = False
outputs = []
for name in layer_name:
    output = unet_model.get_layer(name).output
    outputs.append(output)
perceptual_model = Model(inputs = unet_model.input, outputs = outputs) #get model's output
#define perceptual loss
def compute_perceptual_loss (y_true, y_pred):
    loss = 0.0
    f_true = perceptual_model(y_true)
    f_pred = perceptual_model(y_pred)
    if not isinstance(f_true, (list, tuple)):
        f_true = [f_true]
        f_pred = [f_pred]
    weights = [2.0, 0.5, 0.5]
    #weights = [1.0]*len(f_true)
    for ft, fp, w in zip(f_true, f_pred, weights):
        loss += w * tf.reduce_mean(tf.square(ft - fp))
    return loss
