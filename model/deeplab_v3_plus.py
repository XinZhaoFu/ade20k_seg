from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, add
from model.network_utils import Con_Bn_Act, Sep_Con_Bn_Act


class Deeplab_v3_plus(Model):
    def __init__(self, final_filters, num_middle, img_size=256, input_channel=3, aspp_filters=256,
                 final_activation=None):
        super(Deeplab_v3_plus, self).__init__()
        self.final_filters = final_filters
        self.num_middle = num_middle
        self.img_size = img_size
        self.input_channel = input_channel
        self.aspp_filters = aspp_filters
        self.backbone_low2_filters = 256
        self.final_activation = final_activation

        self.backbone = Xception_BackBone(num_middle=self.num_middle)
        self.aspp = Aspp(filters=self.aspp_filters)
        self.aspp_up = UpSampling2D(size=(4, 4), name='aspp_up')
        self.con_low = Con_Bn_Act(filters=self.backbone_low2_filters, kernel_size=(1, 1), name='con_low')
        self.con_concat = Con_Bn_Act(filters=256, kernel_size=(3, 3), name='con_concat')
        self.up_concat = UpSampling2D(size=(4, 4), name='up_concat')
        self.out_con = Con_Bn_Act(filters=self.final_filters, kernel_size=(3, 3),
                                  activation=self.final_activation, name='out')

    def call(self, inputs):
        backbone_low2, backbone_out = self.backbone(inputs)
        aspp = self.aspp(backbone_out)
        aspp_up = self.aspp_up(aspp)

        con_low = self.con_low(backbone_low2)

        concat = concatenate([aspp_up, con_low], axis=3)
        con_concat = self.con_concat(concat)
        up = self.up_concat(con_concat)
        out = self.out_con(up)

        return out


class Xception_BackBone(Model):
    def __init__(self, num_middle):
        super(Xception_BackBone, self).__init__()
        self.num_middle = num_middle

        #   entry flow
        self.entry_con1_1 = Con_Bn_Act(filters=32, name='entry_con1_1')
        self.entry_con1_2 = Con_Bn_Act(filters=64, strides=2, name='entry_con1_2')

        self.entry_con_res_2 = Con_Bn_Act(filters=128, kernel_size=(1, 1), strides=2, name='entry_con_res_2')
        self.entry_sep_con2_1 = Sep_Con_Bn_Act(filters=128, name='entry_sep_con2_1')
        self.entry_sep_con2_2 = Sep_Con_Bn_Act(filters=128, name='entry_sep_con2_2')
        self.entry_sep_con2_3 = Sep_Con_Bn_Act(filters=128, strides=2, name='entry_sep_con2_3')

        self.entry_con_res_3 = Con_Bn_Act(filters=256, kernel_size=(1, 1), strides=2, name='entry_con_res_3')
        self.entry_sep_con3_1 = Sep_Con_Bn_Act(filters=256, name='entry_sep_con3_1')
        self.entry_sep_con3_2 = Sep_Con_Bn_Act(filters=256, name='entry_sep_con3_2')
        self.entry_sep_con3_3 = Sep_Con_Bn_Act(filters=256, strides=2, name='entry_sep_con3_3')

        self.entry_con_res_4 = Con_Bn_Act(filters=256, kernel_size=(1, 1), strides=2, name='entry_con_res_4')
        self.entry_sep_con4_1 = Sep_Con_Bn_Act(filters=256, name='entry_sep_con4_1')
        self.entry_sep_con4_2 = Sep_Con_Bn_Act(filters=256, name='entry_sep_con4_2')
        self.entry_sep_con4_3 = Sep_Con_Bn_Act(filters=256, kernel_size=(3, 3), strides=2, name='entry_sep_con4_3')

        # middle flow
        self.middle_con_res_middle = Con_Bn_Act(filters=256, kernel_size=(1, 1), name='middle_con_res_middle')
        self.middle_sep_con_middle_x3 = Sep_Con_Bn_Act(filters=256, name='middle_sep_con_middle_x3')

        # exit flow
        self.exit_con_res_1 = Con_Bn_Act(filters=256, kernel_size=(1, 1), name='exit_con_res_1')
        self.exit_sep_con1_1 = Sep_Con_Bn_Act(filters=256, name='exit_sep_con1_1')
        self.exit_sep_con1_x2 = Sep_Con_Bn_Act(filters=256, name='exit_sep_con1_x2')

        self.exit_sep_con2_1 = Sep_Con_Bn_Act(filters=256, name='exit_sep_con2_1')
        self.exit_sep_con2_2 = Sep_Con_Bn_Act(filters=256, name='exit_sep_con2_2')
        self.exit_sep_con2_3 = Sep_Con_Bn_Act(filters=256, name='exit_sep_con2_3')

    def call(self, inputs):
        #   entry flow
        con1_1 = self.entry_con1_1(inputs)
        con1_2 = self.entry_con1_2(con1_1)

        con_res_2 = self.entry_con_res_2(con1_2)
        con2_1 = self.entry_sep_con2_1(con1_2)
        con2_2 = self.entry_sep_con2_2(con2_1)
        con2_3 = self.entry_sep_con2_3(con2_2)
        add2 = add([con2_3, con_res_2])

        con_res_3 = self.entry_con_res_3(add2)
        con3_1 = self.entry_sep_con3_1(add2)
        con3_2 = self.entry_sep_con3_2(con3_1)
        con3_3 = self.entry_sep_con3_3(con3_2)
        add3 = add([con3_3, con_res_3])

        con_res_4 = self.entry_con_res_4(add3)
        con4_1 = self.entry_sep_con4_1(add3)
        con4_2 = self.entry_sep_con4_2(con4_1)
        con4_3 = self.entry_sep_con4_3(con4_2)
        add4 = add([con4_3, con_res_4])

        # middle flow
        add_middle = add4
        for _ in range(self.num_middle):
            con_res_middle = self.middle_con_res_middle(add_middle)
            con_middle_1 = self.middle_sep_con_middle_x3(add_middle)
            con_middle_2 = self.middle_sep_con_middle_x3(con_middle_1)
            con_middle_3 = self.middle_sep_con_middle_x3(con_middle_2)
            add_middle = add([con_middle_3, con_res_middle])

        # exit flow
        con_res_e1 = self.exit_con_res_1(add_middle)
        con_e1_1 = self.exit_sep_con1_1(add_middle)
        con_e1_2 = self.exit_sep_con1_x2(con_e1_1)
        con_e1_3 = self.exit_sep_con1_x2(con_e1_2)
        add_e1 = add([con_e1_3, con_res_e1])

        con_e2_1 = self.exit_sep_con2_1(add_e1)
        con_e2_2 = self.exit_sep_con2_2(con_e2_1)
        out = self.exit_sep_con2_3(con_e2_2)

        return [add2, out]


class Aspp(Model):
    def __init__(self, filters=256):
        super(Aspp, self).__init__()
        self.filters = filters

        self.con1x1 = Conv2D(filters=self.filters, kernel_size=(1, 1), padding='same', name='aspp_con1x1')

        self.dila_con1 = Conv2D(filters=self.filters, kernel_size=(3, 3), dilation_rate=2, padding='same',
                                name='aspp_dila_con1')
        self.dila_con2 = Conv2D(filters=self.filters, kernel_size=(3, 3), dilation_rate=3, padding='same',
                                name='aspp_dila_con2')
        self.dila_con3 = Conv2D(filters=self.filters, kernel_size=(3, 3), dilation_rate=4, padding='same',
                                name='aspp_dila_con3')

        self.pooling_1 = MaxPooling2D(name='aspp_pooling_pooling')
        self.pooling_2 = Conv2D(filters=self.filters, kernel_size=(1, 1), padding='same',
                                name='aspp_pooling_con1x1')
        self.pooling_3 = UpSampling2D(name='aspp_pooling_upsampling')

        self.concat_2 = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), padding='same',
                                   name='aspp_concate_con1x1')

    def call(self, inputs):
        con1x1 = self.con1x1(inputs)

        dila_con6x6 = self.dila_con1(inputs)
        dila_con12x12 = self.dila_con2(inputs)
        dila_con18x18 = self.dila_con3(inputs)

        pooling_1 = self.pooling_1(inputs)
        pooling_2 = self.pooling_2(pooling_1)
        pooling_3 = self.pooling_3(pooling_2)

        concat_1 = concatenate([con1x1, dila_con6x6, dila_con12x12, dila_con18x18, pooling_3], axis=3)
        out = self.concat_2(concat_1)

        return out
