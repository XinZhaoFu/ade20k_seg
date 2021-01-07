from tensorflow.keras import Model
from model.network_utils import Con_Bn_Act, DW_Con_Bn_Act
from tensorflow.keras.layers import MaxPooling2D, concatenate, GlobalAveragePooling2D, Activation, \
    AveragePooling2D, UpSampling2D, add


class Detail_Branch(Model):
    def __init__(self, filters=64):
        super(Detail_Branch, self).__init__()
        self.filters = filters

        self.s1_con_1 = Con_Bn_Act(filters=self.filters, strides=2, name='detail_branch_s1_con_1')
        self.s1_con_2 = Con_Bn_Act(filters=self.filters, name='detail_branch_s1_con_2')

        self.s2_con_1 = Con_Bn_Act(filters=self.filters, strides=2, name='detail_branch_s2_con_1')
        self.s2_con_x2 = Con_Bn_Act(filters=self.filters, name='detail_branch_s2_con_x2')

        self.s3_con_1 = Con_Bn_Act(filters=self.filters * 2, strides=2, name='detail_branch_s3_con_1')
        self.s3_con_x2 = Con_Bn_Act(filters=self.filters * 2, name='detail_branch_s3_con_x2')

    def call(self, inputs):
        s1_con_1 = self.s1_con_1(inputs)
        s1_con_2 = self.s1_con_2(s1_con_1)

        s2_con_1 = self.s2_con_1(s1_con_2)
        s2_con_2 = self.s2_con_x2(s2_con_1)
        s2_con_3 = self.s2_con_x2(s2_con_2)

        s3_con_1 = self.s3_con_1(s2_con_3)
        s3_con_2 = self.s3_con_x2(s3_con_1)
        out = self.s3_con_x2(s3_con_2)

        return out


class Stem_Block(Model):
    def __init__(self, filters=16):
        super(Stem_Block, self).__init__()
        self.filters = filters

        self.con_1 = Con_Bn_Act(filters=self.filters, strides=2, name='stem_block_con_1')

        self.branch1_con_1 = Con_Bn_Act(kernel_size=(1, 1), filters=self.filters, name='stem_block_branch1_con_1')
        self.branch1_con_2 = Con_Bn_Act(filters=self.filters, strides=2, name='stem_block_branch1_con_2')

        self.branch2_maxpooling = MaxPooling2D(strides=2, name='stem_block_branch2_maxpooling')

        self.concat_con = Con_Bn_Act(filters=self.filters, name='stem_block_concat_con')

    def call(self, inputs):
        con_1 = self.con_1(inputs)

        branch_1_con_1 = self.branch1_con_1(con_1)
        branch_1_con_2 = self.branch1_con_2(branch_1_con_1)

        branch_2_maxpooling = self.branch2_maxpooling(con_1)

        concat = concatenate([branch_1_con_2, branch_2_maxpooling], axis=3)
        out = self.concat_con(concat)

        return out


class Context_Embedding_Block(Model):
    def __init__(self, filters=128):
        super(Context_Embedding_Block, self).__init__()
        self.filters = filters

        self.gapooling = GlobalAveragePooling2D(pool_size=(3, 3), name='context_embedding_block_gapooling')
        self.con_1x1 = Con_Bn_Act(kernel_size=(1, 1), filters=self.filters, name='context_embedding_block_con_1x1')

        self.add_con_2 = Con_Bn_Act(filters=self.filters, name='context_embedding_block_concat_con')

    def call(self, inputs):
        gapooling = self.gapooling(inputs)
        con_1x1 = self.con_1x1(gapooling)

        add_1 = add([inputs, con_1x1])
        out = self.add_con_2(add_1)

        return out


class Gather_Expansion_Down_Block(Model):
    def __init__(self, filters):
        super(Gather_Expansion_Down_Block, self).__init__()
        self.filters = filters

        self.con_3x3 = Con_Bn_Act(filters=self.filters)
        self.dw_con_3x3_1 = DW_Con_Bn_Act(filters=self.filters*6, strides=2, activation=None)
        self.dw_con_3x3_2 = DW_Con_Bn_Act(filters=self.filters*6, activation=None)
        self.con_1x1 = Con_Bn_Act(kernel_size=(1, 1), filters=self.filters)

        self.res_dw_con_3x3 = DW_Con_Bn_Act(filters=self.filters, strides=2, activation=None)
        self.res_con_1x1 = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1))

        self.relu = Activation('relu')

    def call(self, inputs):
        con_3x3 = self.con_3x3(inputs)
        dw_con_3x3_1 = self.dw_con_3x3_1(con_3x3)
        dw_con_3x3_2 = self.dw_con_3x3_2(dw_con_3x3_1)
        con_1x1 = self.con_1x1(dw_con_3x3_2)

        res_sw_con_3x3 = self.res_dw_con_3x3(inputs)
        res_con_1x1 = self.res_con_1x1(res_sw_con_3x3)

        add_res = add([con_1x1, res_con_1x1])
        out = self.relu(add_res)

        return out


class Gather_Expansion_Block(Model):
    def __init__(self, filters):
        super(Gather_Expansion_Block, self).__init__()
        self.filters = filters

        self.con_3x3 = Con_Bn_Act(filters=self.filters)
        self.dw_con_3x3 = DW_Con_Bn_Act(filters=self.filters*6, activation=None)
        self.con_1x1 = Con_Bn_Act(kernel_size=(1, 1), filters=self.filters)

        self.relu = Activation('relu')

    def call(self, inputs):
        con_3x3 = self.con_3x3(inputs)
        dw_con_3x3 = self.dw_con_3x3_1(con_3x3)
        con_1x1 = self.con_1x1(dw_con_3x3)

        add_res = add([con_1x1, inputs])
        out = self.relu(add_res)

        return out


class Bilateral_Guided_Aggregation_Block(Model):
    def __init__(self, filters):
        super(Bilateral_Guided_Aggregation_Block, self).__init__()
        self.filters = filters

        self.detail_branch_1_1_dw_con_3x3 = DW_Con_Bn_Act(filters=self.filters, activation=None)
        self.detail_branch_1_2_con_1x1 = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1))

        self.detail_branch_2_1_con3x3 = Con_Bn_Act(filters=self.filters, strides=2, activation=None)
        self.detail_branch_2_2_apooling = AveragePooling2D(pool_size=(3, 3), strides=2)

        self.semantic_branch_1_1_con_3x3 = Con_Bn_Act(filters=self.filters, activation=None)
        self.semantic_branch_1_2_up_4x4 = UpSampling2D(size=(4, 4))
        self.semantic_branch_1_3_sigmoid = Activation('sigmoid')

        self.semantic_branch_2_1_dw_con_3x3 = DW_Con_Bn_Act(filters=self.filters, activation=None)
        self.semantic_branch_2_2_con_1x1 = Con_Bn_Act(kernel_size=(1, 1), filters=self.filters)
        self.semantic_branch_2_3_sigmoid = Activation('sigmoid')

