from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D
from tensorflow.keras import Model


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


class Con_Bn_Act(Model):
    def __init__(self, filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu'):
        super(Con_Bn_Act, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation

        self.con = Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                          padding=self.padding, strides=self.strides, use_bias=False)
        self.bn = BatchNormalization()
        self.act = Activation(self.activation)

    def call(self, inputs):
        con = self.con(inputs)
        bn = self.bn(con)
        if self.kernel_size != (1, 1):
            out = self.act(bn)
        else:
            out = bn
        return out