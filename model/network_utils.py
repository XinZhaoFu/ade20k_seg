from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras import Model, regularizers


class Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 padding='same',
                 strides=1,
                 activation='relu',
                 dilation_rate=1,
                 name=None,
                 kernel_regularizer=False):
        super(Con_Bn_Act, self).__init__()
        self.kernel_regularizer = kernel_regularizer
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.block_name = name

        if self.kernel_regularizer:
            self.con_regularizer = regularizers.l2()
        else:
            self.con_regularizer = None

        # kernel_initializer_special_cases = ['he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        self.con = Conv2D(filters=self.filters,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          strides=self.strides,
                          use_bias=False,
                          dilation_rate=(self.dilation_rate, self.dilation_rate),
                          name=self.block_name,
                          kernel_regularizer=self.con_regularizer,
                          kernel_initializer='glorot_uniform')
        self.bn = BatchNormalization()
        if self.activation is not None:
            self.act = Activation(self.activation)

    def call(self, inputs):
        con = self.con(inputs)
        bn = self.bn(con)
        if self.kernel_size != (1, 1) and self.activation is not None:
            out = self.act(bn)
        else:
            out = bn
        return out


class Sep_Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 padding='same',
                 strides=1,
                 activation='relu',
                 name=None):
        super(Sep_Con_Bn_Act, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.block_name = name

        self.con = SeparableConv2D(filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   strides=self.strides,
                                   use_bias=False,
                                   name=self.block_name)
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


class DW_Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=1,
                 use_bias=False,
                 padding='same',
                 name=None,
                 activation='relu'):
        super(DW_Con_Bn_Act, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.padding = padding
        self.block_name = name
        self.activation = activation

        self.con_1x1 = Con_Bn_Act(filters=self.filters,
                                  kernel_size=(1, 1))

        self.dw_con = DepthwiseConv2D(kernel_size=self.kernel_size,
                                      strides=self.strides,
                                      use_bias=self.use_bias,
                                      padding=self.padding,
                                      name=self.block_name)
        self.bn = BatchNormalization()
        if self.activation is not None:
            self.act = Activation(self.activation)

    def call(self, inputs):
        con_1x1 = self.con_1x1(inputs)

        con = self.dw_con(con_1x1)
        bn = self.bn(con)

        if self.kernel_size != (1, 1) and self.activation is not None:
            out = self.act(bn)
        else:
            out = bn
        return out
