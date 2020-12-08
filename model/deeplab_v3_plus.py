from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, \
    UpSampling2D, concatenate, SeparableConv2D, add


class Deeplab_v3_plus(Model):
    def __init__(self, num_class):
        super(Deeplab_v3_plus).__init__()
        self.num_class = num_class

    def __call__(self, inputs):
        add2, back_bone_out = self._xception_back_bone(inputs)

        aspp = self._aspp(back_bone_out)
        aspp_up = UpSampling2D(size=(4, 4))(aspp)
        con_add2 = self._conv_bn_activation(inputs=add2, filters=256, kernel_size=(1, 1))
        concat = concatenate([aspp_up, con_add2], axis=3)

        con_concat = self._conv_bn_activation(inputs=concat, filters=256, kernel_size=(3, 3))
        up = UpSampling2D(size=(4, 4))(con_concat)

        out = self._conv_bn_activation(inputs=up, filters=self.num_class, kernel_size=(3, 3), activation='softmax')

        return out

    def _xception_back_bone(self, inputs, num_middle=16):
        #   entry flow
        con1_1 = self._conv_bn_activation(inputs=inputs, filters=32, kernel_size=(3, 3), strides=2)
        con1_2 = self._conv_bn_activation(inputs=con1_1, filters=64, kernel_size=(3, 3))

        con_res_2 = self._conv_bn_activation(inputs=con1_2, filters=128, kernel_size=(1, 1), strides=2)
        con2_1 = self._sep_conv_bn_activation(inputs=con1_2, filters=128, kernel_size=(3, 3))
        con2_2 = self._sep_conv_bn_activation(inputs=con2_1, filters=128, kernel_size=(3, 3))
        con2_3 = self._sep_conv_bn_activation(inputs=con2_2, filters=128, kernel_size=(3, 3), strides=2)
        add2 = add([con2_3, con_res_2])

        con_res_3 = self._conv_bn_activation(inputs=add2, filters=256, kernel_size=(1, 1), strides=2)
        con3_1 = self._sep_conv_bn_activation(inputs=add2, filters=256, kernel_size=(3, 3))
        con3_2 = self._sep_conv_bn_activation(inputs=con3_1, filters=256, kernel_size=(3, 3))
        con3_3 = self._sep_conv_bn_activation(inputs=con3_2, filters=256, kernel_size=(3, 3), strides=2)
        add3 = add([con3_3, con_res_3])

        con_res_4 = self._conv_bn_activation(inputs=add3, filters=728, kernel_size=(1, 1), strides=2)
        con4_1 = self._sep_conv_bn_activation(inputs=add3, filters=728, kernel_size=(3, 3))
        con4_2 = self._sep_conv_bn_activation(inputs=con4_1, filters=728, kernel_size=(3, 3))
        con4_3 = self._sep_conv_bn_activation(inputs=con4_2, filters=728, kernel_size=(3, 3), strides=2)
        add4 = add([con4_3, con_res_4])

        # middle flow
        add_middle = add4
        for _ in range(num_middle):
            con_res_middle = self._conv_bn_activation(inputs=add_middle, filters=728, kernel_size=(1, 1), strides=2)
            con_middle_1 = self._sep_conv_bn_activation(inputs=add_middle, filters=728, kernel_size=(3, 3))
            con_middle_2 = self._sep_conv_bn_activation(inputs=con_middle_1, filters=728, kernel_size=(3, 3))
            con_middle_3 = self._sep_conv_bn_activation(inputs=con_middle_2, filters=728, kernel_size=(3, 3))
            add_middle = add([con_middle_3, con_res_middle])

        # exit flow
        con_res_21 = self._conv_bn_activation(inputs=add_middle, filters=1024, kernel_size=(1, 1), strides=2)
        con21_1 = self._sep_conv_bn_activation(inputs=add_middle, filters=728, kernel_size=(3, 3))
        con21_2 = self._sep_conv_bn_activation(inputs=con21_1, filters=1024, kernel_size=(3, 3))
        con21_3 = self._sep_conv_bn_activation(inputs=con21_2, filters=1024, kernel_size=(3, 3), strides=2)
        add21 = add([con21_3, con_res_21])

        con22_1 = self._sep_conv_bn_activation(inputs=add21, filters=1536, kernel_size=(3, 3))
        con22_2 = self._sep_conv_bn_activation(inputs=con22_1, filters=1536, kernel_size=(3, 3))
        out = self._sep_conv_bn_activation(inputs=con22_2, filters=2048, kernel_size=(3, 3))

        return [add2, out]

    def _sep_conv_bn_activation(self, inputs, filters, kernel_size=(3, 3), strides=1, padding='same',
                                activation='relu'):
        conv = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, use_bias=False)(inputs)
        bn = BatchNormalization()(conv)
        if kernel_size == (1, 1):
            out = bn
        else:
            out = Activation(activation=activation)(bn)

        return out

    def _conv_bn_activation(self, inputs, filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu'):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding=padding, use_bias=False)(inputs)
        bn = BatchNormalization()(conv)
        if kernel_size == (1, 1):
            out = bn
        else:
            out = Activation(activation=activation)(bn)

        return out

    def _aspp(self, inputs, filters=256):
        con1x1 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(inputs)

        dila_con6x6 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=6)(inputs)
        dila_con12x12 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=12)(inputs)
        dila_con18x18 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=18)(inputs)

        pooling_1 = MaxPooling2D()(inputs)
        pooling_2 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(pooling_1)
        pooling_3 = UpSampling2D()(pooling_2)

        concat_1 = concatenate([con1x1, dila_con6x6, dila_con12x12, dila_con18x18, pooling_3], axis=3)
        concat_2 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=False)(concat_1)
        out = BatchNormalization()(concat_2)

        return out
