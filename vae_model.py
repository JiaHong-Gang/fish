from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2DTranspose
from  tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D,Concatenate,Lambda, Reshape
from  tensorflow.keras.layers import Input, Add, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K
from config import channel_num

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape = tf.shape(z_mean))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    return z

def vae(input_shape = (512, 512,3), latent_dim = 64):
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
    z_log_var = Lambda(lambda x: tf.clip_by_value(x, -10.0, 10.0), name="clip_logvar")(z_log_var)
    # Sampling layer
    z = Lambda(sampling, output_shape = (latent_dim,), name = "z")([z_mean, z_log_var])
    h, w, c = K.int_shape(lp4)[1:]
    flat_dim = h * w * c
    # decoder
    decoder_input = Dense(flat_dim, activation = "relu", name = "decoder_input")(z)
    reshape =Reshape((h, w,c))(decoder_input)
    tc4 = Conv2DTranspose(filters=256, kernel_size= 2, strides = 2, padding="same", name="transpose_conv4")(reshape)
    rc4 = Conv2D(filters=256, kernel_size= 3, activation= "relu", padding="same", name="right_conv4_1")(tc4)
   # rc4 = Conv2D(filters=512, kernel_size=3, activation="relu", padding="same", name="right_conv4_2")(rc4)

    tc3 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="same", name="transpose_conv3")(rc4)
    rc3 = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", name="right_conv3_1")(tc3)
   # rc3 = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", name="right_conv3_2")(rc3)

    tc2 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="same", name="transpose_conv2")(rc3)
    rc2 = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", name="right_conv2_1")(tc2)
    #rc2 = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same", name="right_conv2_2")(rc2)

    tc1 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="same", name="transpose_conv1")(rc2)
    rc1 = Conv2D(filters=32, kernel_size=3, activation="relu", padding="same", name="right_conv1_1")(tc1)
    #rc1= Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", name="right_conv1_2")(rc1)

    output_layer = Conv2D(3, kernel_size = 1, activation = "sigmoid",name = "output_layer")(rc1)

    model = Model(inputs = input_layer,outputs = [output_layer, z_mean, z_log_var], name = "vae_unet_model")

    return model
