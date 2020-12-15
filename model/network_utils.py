from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D
from tensorflow.keras import Model, regularizers


class Con_Bn_Act(Model):
    def __init__(self, filters, img_size, input_channel,  kernel_size=(3, 3), padding='same', strides=1,
                 activation='relu', dilation_rate=1, name=None):
        super(Con_Bn_Act, self).__init__()
        self.filters = filters
        self.img_size = img_size
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.block_name = name

        self.con = Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding,
                          strides=self.strides, use_bias=False, dilation_rate=(self.dilation_rate, self.dilation_rate),
                          input_shape=(self.img_size, self.img_size, self.input_channel), name=self.block_name,
                          kernel_regularizer=regularizers.l2())
        self.bn = BatchNormalization(input_shape=(self.img_size, self.img_size, self.filters))
        self.act = Activation(self.activation)

    def call(self, inputs):
        con = self.con(inputs)
        bn = self.bn(con)
        if self.kernel_size != (1, 1):
            out = self.act(bn)
        else:
            out = bn
        return out


class Sep_Con_Bn_Act(Model):
    def __init__(self, filters, input_channel, img_size, kernel_size=(3, 3), padding='same', strides=1,
                 activation='relu', name=None):
        super(Sep_Con_Bn_Act, self).__init__()
        self.filters = filters
        self.img_size = img_size
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.block_name = name

        self.con = SeparableConv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding,
                               strides=self.strides, use_bias=False, name=self.block_name,
                                   input_shape=(self.img_size, self.img_size, self.input_channel),
                                   kernel_regularizer=regularizers.l2())
        self.bn = BatchNormalization(input_shape=(self.img_size, self.img_size, self.filters))
        self.act = Activation(self.activation)

    def call(self, inputs):
        con = self.con(inputs)
        bn = self.bn(con)
        if self.kernel_size != (1, 1):
            out = self.act(bn)
        else:
            out = bn
        return out
