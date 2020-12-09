from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D


def sep_conv_bn_activation(inputs, filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu'):
    conv = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding=padding, use_bias=False)(inputs)
    bn = BatchNormalization()(conv)
    if kernel_size == (1, 1):
        out = bn
    else:
        out = Activation(activation=activation)(bn)

    return out


def conv_bn_activation(inputs, filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu'):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                  padding=padding, use_bias=False)(inputs)
    bn = BatchNormalization()(conv)
    if kernel_size == (1, 1):
        out = bn
    else:
        out = Activation(activation=activation)(bn)

    return out