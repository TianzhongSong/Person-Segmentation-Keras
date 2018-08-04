from keras.layers import Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.initializers import orthogonal, he_normal
from keras.regularizers import l2


def SegNet(nClasses, input_height=256, input_width=256):
    img_input = Input(shape=(input_height, input_width, 3))
    kernel_size = 3
    # encoder
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 128x128
    x = Conv2D(128, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 64x64
    x = Conv2D(256, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 32x32
    x = Conv2D(512, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 16x16
    x = Conv2D(512, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 8x8

    # decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = Conv2D(nClasses, (1, 1), padding='valid',
               kernel_initializer=he_normal(), kernel_regularizer=l2(0.005))(x)
    x = Reshape((nClasses, input_height * input_width))(x)
    x = Permute((2, 1))(x)

    x = Activation('softmax')(x)
    model = Model(img_input, x, name='SegNet')
    return model