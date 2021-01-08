from tensorflow.keras import Model
from model.network_utils import Con_Bn_Act, DW_Con_Bn_Act
from tensorflow.keras.layers import MaxPooling2D, concatenate, GlobalAveragePooling2D, Activation, \
    AveragePooling2D, UpSampling2D, add, multiply


class BisenetV2(Model):
    def __init__(self, detail_filters=64,
                 semantic_filters=None,
                 aggregation_filters=128,
                 final_filters=151,
                 final_act='softmax'):
        super(BisenetV2, self).__init__()
        self.final_act = final_act
        self.final_filters = final_filters
        if semantic_filters is None:
            semantic_filters = [16, 32, 64, 128]
        self.aggregation_filters = aggregation_filters
        self.semantic_filters = semantic_filters
        self.detail_filters = detail_filters

        self.detail_branch = Detail_Branch(filters=64)
        self.semantic_branch = Semantic_Branch(filters=[16, 32, 64, 128])
        self.aggregation = Bilateral_Guided_Aggregation_Block(filters=128,
                                                              final_filters=151,
                                                              final_act='softmax')

    def call(self, inputs):
        detail_branch = self.detail_branch(inputs)
        semantic_branch = self.semantic_branch(inputs)
        out = self.aggregation([detail_branch, semantic_branch])

        return out


class Detail_Branch(Model):
    def __init__(self, filters=64):
        super(Detail_Branch, self).__init__()
        self.filters = filters

        self.s1_con_1 = Con_Bn_Act(filters=self.filters,
                                   strides=2,
                                   name='detail_branch_s1_con_1')
        self.s1_con_2 = Con_Bn_Act(filters=self.filters,
                                   name='detail_branch_s1_con_2')

        self.s2_con_1 = Con_Bn_Act(filters=self.filters,
                                   strides=2,
                                   name='detail_branch_s2_con_1')
        self.s2_con_x2 = Con_Bn_Act(filters=self.filters,
                                    name='detail_branch_s2_con_x2')

        self.s3_con_1 = Con_Bn_Act(filters=self.filters * 2,
                                   strides=2,
                                   name='detail_branch_s3_con_1')
        self.s3_con_x2 = Con_Bn_Act(filters=self.filters * 2,
                                    name='detail_branch_s3_con_x2')

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


class Semantic_Branch(Model):
    def __init__(self, filters=None):
        super(Semantic_Branch, self).__init__()
        if filters is None:
            filters = [16, 32, 64, 128]
        self.filters = filters

        self.stem = Stem_Block(filters=self.filters[0])

        self.s3_GE_down_1 = Gather_Expansion_Down_Block(filters=self.filters[1])
        self.s3_GE_down_2 = Gather_Expansion_Down_Block(filters=self.filters[1])
        self.s3_GE_3 = Gather_Expansion_Block(filters=self.filters[1])

        self.s4_GE_down_1 = Gather_Expansion_Down_Block(filters=self.filters[2])
        self.s4_GE_down_2 = Gather_Expansion_Down_Block(filters=self.filters[2])
        self.s4_GE_3 = Gather_Expansion_Block(filters=self.filters[2])

        self.s5_GE_down_1 = Gather_Expansion_Down_Block(filters=self.filters[3])
        self.s5_GE_x3 = Gather_Expansion_Block(filters=self.filters[3])

        self.s5_CE = Context_Embedding_Block(filters=self.filters[3])

    def call(self, inputs):
        stem = self.stem(inputs)

        s3_GE_down_1 = self.s3_GE_down_1(stem)
        s3_GE_down_2 = self.s3_GE_down_2(s3_GE_down_1)
        s3_GE_3 = self.s3_GE_3(s3_GE_down_2)

        s4_GE_down_1 = self.s4_GE_down_1(s3_GE_3)
        s4_GE_down_2 = self.s4_GE_down_2(s4_GE_down_1)
        s4_GE_3 = self.s4_GE_3(s4_GE_down_2)

        s5_GE_down_1 = self.s5_GE_down_1(s4_GE_3)
        s5_GE_2 = self.s5_GE_x3(s5_GE_down_1)
        s5_GE_3 = self.s5_GE_x3(s5_GE_2)
        s5_GE_4 = self.s5_GE_x3(s5_GE_3)

        out = self.s5_CE(s5_GE_4)

        return out


class Stem_Block(Model):
    def __init__(self, filters=16):
        super(Stem_Block, self).__init__()
        self.filters = filters

        self.con_1 = Con_Bn_Act(filters=self.filters,
                                strides=2,
                                name='stem_block_con_1')

        self.branch1_con_1 = Con_Bn_Act(kernel_size=(1, 1),
                                        filters=self.filters,
                                        name='stem_block_branch1_con_1')
        self.branch1_con_2 = Con_Bn_Act(filters=self.filters,
                                        strides=2,
                                        name='stem_block_branch1_con_2')

        self.branch2_maxpooling = MaxPooling2D(strides=2,
                                               name='stem_block_branch2_maxpooling')

        self.concat_con = Con_Bn_Act(filters=self.filters,
                                     name='stem_block_concat_con')

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

        self.gapooling = GlobalAveragePooling2D(pool_size=(3, 3),
                                                name='context_embedding_block_gapooling')
        self.con_1x1 = Con_Bn_Act(kernel_size=(1, 1),
                                  filters=self.filters,
                                  name='context_embedding_block_con_1x1')

        self.add_con_2 = Con_Bn_Act(filters=self.filters,
                                    name='context_embedding_block_concat_con')

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

        self.con_3x3 = Con_Bn_Act(filters=self.filters,
                                  name='gather_expansion_down_con_3x3')
        self.dw_con_3x3_1 = DW_Con_Bn_Act(filters=self.filters*6,
                                          strides=2,
                                          activation=None,
                                          name='gather_expansion_down_dw_con_3x3_1')
        self.dw_con_3x3_2 = DW_Con_Bn_Act(filters=self.filters*6,
                                          activation=None,
                                          name='gather_expansion_down_dw_con_3x3_2')
        self.con_1x1 = Con_Bn_Act(kernel_size=(1, 1),
                                  filters=self.filters,
                                  name='gather_expansion_down_con_1x1')

        self.res_dw_con_3x3 = DW_Con_Bn_Act(filters=self.filters,
                                            strides=2,
                                            activation=None,
                                            name='gather_expansion_down_res_dw_con_3x3')
        self.res_con_1x1 = Con_Bn_Act(filters=self.filters,
                                      kernel_size=(1, 1),
                                      name='gather_expansion_down_res_con_1x1')

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

        self.con_3x3 = Con_Bn_Act(filters=self.filters,
                                  name='gather_expansion_con_3x3')
        self.dw_con_3x3 = DW_Con_Bn_Act(filters=self.filters*6,
                                        activation=None,
                                        name='gather_expansion_dw_con_3x3')
        self.con_1x1 = Con_Bn_Act(kernel_size=(1, 1),
                                  filters=self.filters,
                                  name='gather_expansion_con_1x1')

        self.relu = Activation('relu')

    def call(self, inputs):
        con_3x3 = self.con_3x3(inputs)
        dw_con_3x3 = self.dw_con_3x3_1(con_3x3)
        con_1x1 = self.con_1x1(dw_con_3x3)

        add_res = add([con_1x1, inputs])
        out = self.relu(add_res)

        return out


class Bilateral_Guided_Aggregation_Block(Model):
    def __init__(self, filters=128, final_filters=151, final_act='softmax'):
        super(Bilateral_Guided_Aggregation_Block, self).__init__()
        self.final_act = final_act
        self.final_filters = final_filters
        self.filters = filters

        self.detail_remain_1_dw_con_3x3 = DW_Con_Bn_Act(filters=self.filters,
                                                        activation=None,
                                                        name='aggregation_detail_remain_1_dw_con_3x3')
        self.detail_remain_2_con_1x1 = Con_Bn_Act(filters=self.filters,
                                                  kernel_size=(1, 1),
                                                  name='aggregation_detail_remain_2_con_1x1')

        self.detail_down_1_con_3x3 = Con_Bn_Act(filters=self.filters,
                                                strides=2,
                                                activation=None,
                                                name='aggregation_detail_down_1_con3x3')
        self.detail_down_2_apooling = AveragePooling2D(pool_size=(3, 3),
                                                       strides=2,
                                                       name='aggregation_detail_down_2_apooling')

        self.semantic_up_1_con_3x3 = Con_Bn_Act(filters=self.filters,
                                                activation=None,
                                                name='aggregation_semantic_up_1_con_3x3')
        self.semantic_up_2_up_4x4 = UpSampling2D(size=(4, 4))
        self.semantic_up_3_sigmoid = Activation('sigmoid')

        self.semantic_remain_1_dw_con_3x3 = DW_Con_Bn_Act(filters=self.filters,
                                                          activation=None,
                                                          name='aggregation_semantic_remain_1_dw_con_3x3')
        self.semantic_remain_2_con_1x1 = Con_Bn_Act(kernel_size=(1, 1),
                                                    filters=self.filters,
                                                    name='aggregation_semantic_remain_2_con_1x1')
        self.semantic_remain_3_sigmoid = Activation('sigmoid')

        self.semantic_up = UpSampling2D(size=(4, 4))

        self.sum_con_3x3 = Con_Bn_Act(filters=self.final_filters,
                                      activation=self.final_act,
                                      name='aggregation_sum_con_3x3')

    def call(self, inputs):
        detail_branch_remain_1_dw_con_3x3 = self.detail_remain_1_dw_con_3x3(inputs[0])
        detail_branch_remain_2_con_1x1 = self.detail_remain_2_con_1x1(detail_branch_remain_1_dw_con_3x3)

        detail_branch_down_1_con3x3 = self.detail_down_1_con_3x3(inputs[0])
        detail_branch_down_2_apooling = self.detail_down_2_apooling(detail_branch_down_1_con3x3)

        semantic_branch_up_1_con_3x3 = self.semantic_up_1_con_3x3(inputs[1])
        semantic_branch_up_2_up_4x4 = self.semantic_up_2_up_4x4(semantic_branch_up_1_con_3x3)
        semantic_branch_up_3_sigmoid = self.semantic_up_3_sigmoid(semantic_branch_up_2_up_4x4)

        semantic_branch_remain_1_dw_con_3x3 = self.semantic_remain_1_dw_con_3x3(inputs[1])
        semantic_branch_remain_2_con_1x1 = self.semantic_remain_2_con_1x1(
            semantic_branch_remain_1_dw_con_3x3)
        semantic_branch_remain_3_sigmoid = self.semantic_remain_3_sigmoid(semantic_branch_remain_2_con_1x1)

        detail_multiply = multiply([detail_branch_remain_2_con_1x1, semantic_branch_up_3_sigmoid])
        semantic_multiply = multiply([semantic_branch_remain_3_sigmoid, detail_branch_down_2_apooling])
        semantic_up = self.semantic_up(semantic_multiply)

        detail_semantic_sum = add([detail_multiply, semantic_up])
        out = self.sum_con_3x3(detail_semantic_sum)

        return out
