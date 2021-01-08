import argparse
import datetime
from model.deeplab_v3_plus import Deeplab_v3_plus
import tensorflow as tf
import os
from data_utils.data_loader_hdf5 import Data_Loader_Hdf5
from data_utils.data_loader_file import Data_Loader_File
from model.unet import UNet_seg
from model.bisenetv2 import BisenetV2
from utils import print_cost_time
import setproctitle
import numpy as np

# 便于他人知道是福福仔的程序在占着茅坑不拉屎
setproctitle.setproctitle("xzf_ade")


def parseArgs():
    """
    获得参数
    :return:
    """
    parser = argparse.ArgumentParser(description='ade20k segmentation demo')
    parser.add_argument('--img_size',
                        dest='img_size',
                        help='img_size',
                        default=512,
                        type=int)
    parser.add_argument('--mask_size',
                        dest='mask_size',
                        help='mask_size',
                        default=64,
                        type=int)
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='learning_rate',
                        default=0,
                        type=float)
    parser.add_argument('--epochs',
                        dest='epochs',
                        help='epochs',
                        default=1,
                        type=int)
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='batch_size',
                        default=8,
                        type=int)
    parser.add_argument('--load_train_file_number',
                        dest='load_train_file_number',
                        help='load_train_file_number',
                        default=20210,
                        type=int)
    parser.add_argument('--load_val_file_number',
                        dest='load_val_file_number',
                        help='load_val_file_number',
                        default=2000,
                        type=int)
    parser.add_argument('--load_file_mode',
                        dest='load_file_mode',
                        help='load_file_mode type is string part or all',
                        default='part',
                        type=str)
    parser.add_argument('--load_data_mode',
                        dest='load_data_mode',
                        help='load_data_mode type is string hdf5 or file',
                        default='file',
                        type=str)
    parser.add_argument('--load_weights',
                        dest='load_weights',
                        help='load_weights type is boolean',
                        default=False, type=bool)
    parser.add_argument('--rewrite_hdf5',
                        dest='rewrite_hdf5',
                        help='rewrite_hdf5 type is boolean',
                        default=False,
                        type=bool)
    parser.add_argument('--data_augmentation',
                        dest='data_augmentation',
                        help='data_augmentation type is boolean',
                        default=False,
                        type=bool)
    args = parser.parse_args()
    return args


class seg_train:
    def __init__(self,
                 load_weights=False,
                 batch_size=8,
                 epochs=0,
                 load_data_mode='hdf5',
                 img_size=512,
                 mask_size=64,
                 load_file_mode='part',
                 load_train_file_number=1000,
                 load_val_file_number=200,
                 rewrite_hdf5=False,
                 data_augmentation=False,
                 augmentation_rate=4,
                 erase_rate=0.1,
                 learning_rate=0,
                 model_name='bisenetv2'):
        self.mask_size = mask_size
        self.model_name = model_name
        self.load_weights = load_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.load_data_mode = load_data_mode
        self.img_size = img_size
        self.load_file_mode = load_file_mode
        self.rewrite_hdf5 = rewrite_hdf5
        self.data_augmentation = data_augmentation
        self.load_train_file_number = load_train_file_number
        self.load_val_file_number = load_val_file_number
        self.erase_rate = erase_rate
        self.augmentation_rate = augmentation_rate
        self.learning_rate = learning_rate
        self.checkpoint_save_path = './checkpoint/' + self.model_name + '_demo1.ckpt'

        self.strategy = tf.distribute.MirroredStrategy()
        print('目前使用gpu数量为: {}'.format(self.strategy.num_replicas_in_sync))
        if self.strategy.num_replicas_in_sync >= 8:
            print('[INFO]----------卡数上八!!!---------')

        if self.load_data_mode == 'hdf5':
            #   load_file_mode部分数据为part 便于测试 全部数据为all 其实也可以随便写 if part else all
            data_loader = Data_Loader_Hdf5(load_file_mode=self.load_file_mode,
                                           mask_size=self.img_size,
                                           rewrite_hdf5=self.rewrite_hdf5,
                                           data_augmentation=self.data_augmentation,
                                           augmentation_rate=self.augmentation_rate,
                                           erase_rate=self.erase_rate)
            self.train_img, self.train_label = data_loader.load_train_data()
            self.val_img, self.val_label = data_loader.load_val_data()
        else:
            data_loader = Data_Loader_File(img_size=self.img_size,
                                           mask_size=self.mask_size,
                                           data_augmentation=False,
                                           batch_size=self.batch_size)
            self.train_datasets = data_loader.load_train_data(load_file_number=self.load_train_file_number)
            self.val_datasets = data_loader.load_val_data(load_file_number=self.load_val_file_number)

    def model_train(self):
        """
        可多卡训练
        :return:
        """
        with self.strategy.scope():
            model = BisenetV2(detail_filters=64,
                              semantic_filters=16,
                              aggregation_filters=128,
                              final_filters=151,
                              final_act='softmax')
            if self.model_name == 'unet':
                model = UNet_seg(filters=128,
                                 img_width=256,
                                 input_channel=3,
                                 num_class=151,
                                 num_con_unit=2)
            if self.model_name == 'deeplabv3plus':
                model = Deeplab_v3_plus(final_filters=151,
                                        num_middle=8,
                                        input_channel=3,
                                        aspp_filters=256,
                                        final_activation='softmax')

            if self.learning_rate > 0:
                print('使用sgd,其值为：\t'.join(str(self.learning_rate)))
                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )
            else:
                print('使用adam')
                model.compile(
                    optimizer='Adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )

            if os.path.exists(self.checkpoint_save_path + '.index') and self.load_weights:
                print("[INFO] loading weights---------怕眼瞎看不见加长版--------")
                model.load_weights(self.checkpoint_save_path)

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_save_path,
                monitor='val_loss',
                save_weights_only=True,
                save_best_only=True,
                mode='auto',
                save_freq='epoch')

        if self.load_data_mode == 'hdf5':
            history = model.fit(
                self.train_img, self.train_label,
                epochs=self.epochs,
                verbose=1,
                batch_size=self.batch_size,
                validation_data=(self.val_img, self.val_label),
                validation_freq=1,
                callbacks=[checkpoint_callback])
        else:
            history = model.fit(
                self.train_datasets,
                epochs=self.epochs,
                verbose=1,
                validation_data=self.val_datasets,
                validation_freq=1,
                callbacks=[checkpoint_callback])

        model.summary()


def main():
    start_time = datetime.datetime.now()

    args = parseArgs()
    seg = seg_train(load_weights=args.load_weights,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    load_data_mode=args.load_data_mode,
                    img_size=args.img_size,
                    mask_size=args.mask_size,
                    load_file_mode=args.load_file_mode,
                    load_train_file_number=args.load_train_file_number,
                    load_val_file_number=args.load_val_file_number,
                    rewrite_hdf5=args.rewrite_hdf5,
                    data_augmentation=args.data_augmentation,
                    learning_rate=args.learning_rate)
    seg.model_train()

    print_cost_time(start_time)


if __name__ == '__main__':
    main()
