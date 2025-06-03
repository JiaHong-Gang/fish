from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2DTranspose
from  tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D,Concatenate,Lambda, Reshape
from  tensorflow.keras.layers import Input, Add, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
from config import channel_class

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape = tf.shape(z_mean))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    return z

def vae(input_shape = (1088, 768,3), latent_dim = 256):
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
    flatten = Flatten()(lp4)

    z_mean = Dense(latent_dim, name = "z_mean")(flatten)
    z_log_var = Dense(latent_dim, name = "z_log_var")(flatten)
    # Sampling layer
    z = Lambda(sampling, output_shape = (latent_dim,), name = "z")([z_mean, z_log_var])
    h, w, c = K.int_shape(lp4)[1:]
    flat_dim = h * w * c
    # decoder
    decoder_input = Dense(flat_dim, activation = "relu", name = "decoder_input")(z)
    reshape =Reshape((h, w, c))(decoder_input)
    tc4 = Conv2DTranspose(filters=512, kernel_size= 2, strides = 2, padding="same", name="transpose_conv4")(reshape)
    #tc4 = Concatenate(name = "concatenate4")([tc4,lc4])
    rc4 = Conv2D(filters=512, kernel_size= 3, activation= "relu", padding="same", name="right_conv4_1")(tc4)
    rc4 = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", name="right_conv4_2")(rc4)

    tc3 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="same", name="transpose_conv3")(rc4)
    #tc3 = Concatenate(name = "concatenate3")([tc3,lc3])
    rc3 = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", name="right_conv3_1")(tc3)
    rc3 = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", name="right_conv3_2")(rc3)

    tc2 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="same", name="transpose_conv2")(rc3)
    #tc2 = Concatenate(name = "concatenate2")([tc2,lc2])
    rc2 = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", name="right_conv2_1")(tc2)
    rc2 = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", name="right_conv2_2")(rc2)

    tc1 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="same", name="transpose_conv1")(rc2)
    #tc1 = Concatenate(name = "concatenate1")([tc1,lc1])
    rc1 = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", name="right_conv1_1")(tc1)
    rc1= Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", name="right_conv1_2")(rc1)

    output_layer1 = Conv2D(channel_class, kernel_size = 1, activation = "sigmoid",name = "RGB_output_layer")(rc1)
    output_layer2 = Conv2D(1, kernel_size = 1 , activation = "sigmoid", name = "mask_output_layer")(rc1)

    model = Model(inputs = input_layer,outputs = [output_layer1, output_layer2, z_mean, z_log_var], name = "vae_model")

    return model