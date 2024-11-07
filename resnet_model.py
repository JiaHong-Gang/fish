import tensorflow as tf
from fontTools.unicodedata import block
from keras.src.layers import GlobalAveragePooling2D
from  tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D
from  tensorflow.keras.layers import Input, Add, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from config import num_class

def identy_block(X, filters, kernel_size = 3, block_name = "id_block"):
    X_shortcut = X
    # first convolutional layer
    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = 1, padding = "same", name = block_name + "_conv1")(X)
    X = BatchNormalization(name = block_name + "_bn1")(X)
    X = Activation("relu", name = block_name + "_act1")(X)

    #second convolutional layer
    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = 1,padding = "same", name = block_name + "_conv2")(X)
    X = BatchNormalization(name = block_name + "_bn2")(X)

    # Add shortcut to the output
    X = Add(name = block_name + "_add")([X, X_shortcut])
    X = Activation("relu", name = block_name + "_act2")(X)

    return X

def convolutional_block(X, filters, kernel_size = 3, s = 2, block_name = "conv_block"):
    X_shortcut = X

    #First convolutional layer
    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = s, padding = "same", name = block_name + "_conv1")(X)
    X = BatchNormalization(name = block_name + "_bn1")(X)
    X = Activation("relu", name = block_name + "_act1")(X)

    #Second convolutional layer
    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = 1, padding = "same", name = block_name + "_conv2")(X)
    X = BatchNormalization(name = block_name + "_bn2")(X)

    #X_shortcut branch
    X_shortcut = Conv2D(filters = filters, kernel_size = kernel_size, strides = s, padding = "same", name = block_name + "shortcut_conv")(X_shortcut)
    X_shortcut = BatchNormalization(name = block_name + "_shortcut_bn")(X_shortcut)

    # Add shortcut to the output
    X = Add(name = block_name + "_add")([X, X_shortcut])
    X = Activation("relu", name = block_name + "_act2")(X)
    return X

def ResNet_18(input_shape = (224,224,3)):
    X_input = Input(input_shape, name = "input_layer")
# initial convolution layer
    X = Conv2D(filters = 64, kernel_size = 7, strides = 2, padding = "same", name = "initial_conv")(X_input)
    X = BatchNormalization(name = "initial_bn")(X)
    X = Activation("relu", name = "initial_act")(X)
#ResNet blocks
    X = tf.keras.layers.MaxPooling2D(pool_size = 3, strides = 2, padding = "same", name = "initial_maxpool")(X)
    X = identy_block(X, 64, 3, block_name = "id_block1")
    X = identy_block(X, 64, 3, block_name = "id_block2")
    X = convolutional_block(X, 128, 3, block_name = "conv_block1")
    X = identy_block(X, 128, 3, block_name= "id_block3")
    X = convolutional_block(X, 256, 3, block_name= "conv_block2")
    X = identy_block(X, 256, 3, block_name= "id_block4")
    X = convolutional_block(X, 512, 3, block_name= "conv_block3")
    X = identy_block(X, 512, 3, block_name= "id_block5")
    X = GlobalAveragePooling2D(name = "global_average_pooling2d")(X)
    predictions = Dense(num_class, activation = "softmax", name = "Dense")(X)
    model = Model(inputs = X_input, outputs = predictions, name = "ResNet18")

    return model
