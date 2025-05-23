from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2DTranspose
from  tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D,Concatenate,Lambda, Reshape
from  tensorflow.keras.layers import Input, Add, Flatten, Dropout, BatchNormalization, Activation, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
from config import num_class


def unet(input_shape = (512, 512,3)):
    input_layer = Input(shape = input_shape, name = "input_image")

    #encoder
    lc1 = Conv2D(filters = 64, kernel_size =3, activation = "relu", padding = "same", name = "left_Conv1_1")(input_layer)
    lc1 = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", name="left_Conv1_2")(lc1)
    lp1 = MaxPooling2D(pool_size = 2, name = "left_pooling1")(lc1)

    lc2 = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", name="left_Conv2_1")(lp1)
    lc2 = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", name="left_Conv2_2")(lc2)
    lp2 = MaxPooling2D(pool_size=2, name="left_pooling2")(lc2)

    lc3 = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", name="left_Conv3_1")(lp2)
    lc3 = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", name="left_Conv3_2")(lc3)
    lp3 = MaxPooling2D(pool_size=2, name="left_pooling3")(lc3)

    lc4 = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", name="left_Conv4_1")(lp3)
    lc4 = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", name="left_Conv4_2")(lc4)
    lp4 = MaxPooling2D(pool_size=2, name="left_pooling4")(lc4)

    #bottom
    bc5 = Conv2D(filters=1024, kernel_size=3, activation="relu", padding = "same", name="bottleneck_1")(lp4)
    bc5 = Conv2D(filters=1024, kernel_size=3, activation="relu", padding = "same", name="bottleneck_2")(bc5)

    # decoder
    tc4 = Conv2DTranspose(filters=512, kernel_size= 2, strides = 2, padding="same", name="transpose_conv4")(bc5)
    tc4 = concatenate([tc4, lc4],name="concate_4")
    rc4 = Conv2D(filters=512, kernel_size= 3, activation= "relu", padding="same", name="right_conv4_1")(tc4)
    rc4 = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", name="right_conv4_2")(rc4)

    tc3 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="same", name="transpose_conv3")(rc4)
    tc3 = concatenate([tc3, lc3], name = "concate_3")
    rc3 = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", name="right_conv3_1")(tc3)
    rc3 = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", name="right_conv3_2")(rc3)

    tc2 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="same", name="transpose_conv2")(rc3)
    tc2 = concatenate([tc2, lc2], name = "concate_2")
    rc2 = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", name="right_conv2_1")(tc2)
    rc2 = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", name="right_conv2_2")(rc2)

    tc1 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="same", name="transpose_conv1")(rc2)
    tc1 = concatenate([tc1, lc1], name = "concate_1")
    rc1 = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", name="right_conv1_1")(tc1)
    rc1= Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", name="right_conv1_2")(rc1)

    output_layer = Conv2D(num_class, kernel_size = 1, activation = "sigmoid" ,name = "output_layer")(rc1)

    model = Model(inputs = input_layer,outputs = output_layer, name = "unet_model")

    return model
