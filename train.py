import argparse
import datetime
from model.deeplab_v3_plus import Deeplab_v3_plus
import tensorflow as tf
import os
from data_utils.data_loader_hdf5 import Data_Loader_Hdf5
from data_utils.data_loader_file import Data_Loader_File
from model.unet import UNet_seg
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
    parser.add_argument('--mask_size', dest='mask_size', help='mask_size', default=256, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', help='learning_rate', default=0, type=float)
    parser.add_argument('--epochs', dest='epochs', help='epochs', default=0, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=1, type=int)
    parser.add_argument('--load_train_file_number', dest='load_train_file_number', help='load_train_file_number',
                        default=1000, type=int)
    parser.add_argument('--load_val_file_number', dest='load_val_file_number', help='load_val_file_number',
                        default=200, type=int)
    parser.add_argument('--load_file_mode', dest='load_file_mode',
                        help='load_file_mode type is string part or all', default='part', type=str)
    parser.add_argument('--load_data_mode', dest='load_data_mode',
                        help='load_data_mode type is string hdf5 or file', default='hdf5', type=str)
    parser.add_argument('--load_weights', dest='load_weights',
                        help='load_weights type is boolean', default=False, type=bool)
    parser.add_argument('--rewrite_hdf5', dest='rewrite_hdf5',
                        help='rewrite_hdf5 type is boolean', default=False, type=bool)
    parser.add_argument('--data_augmentation', dest='data_augmentation',
                        help='data_augmentation type is boolean', default=False, type=bool)
    args = parser.parse_args()
    return args


class seg_train:
    def __init__(self, load_weights=False, batch_size=8, epochs=0, load_data_mode='hdf5', mask_size=256,
                 load_file_mode='part', load_train_file_number=1000, load_val_file_number=200,
                 rewrite_hdf5=False, data_augmentation=False, augmentation_rate=1, erase_rate=0.1,
                 learning_rate=0):
        self.load_weights = load_weights
        # self.checkpoint_save_path = './checkpoint/unet_demo1.ckpt'
        self.checkpoint_save_path = './checkpoint/deeplabv3plus_demo1.ckpt'
        self.batch_size = batch_size
        self.epochs = epochs
        self.load_data_mode = load_data_mode
        self.mask_size = mask_size
        self.load_file_mode = load_file_mode
        self.rewrite_hdf5 = rewrite_hdf5
        self.data_augmentation = data_augmentation
        self.load_train_file_number = load_train_file_number
        self.load_val_file_number = load_val_file_number
        self.erase_rate = erase_rate
        self.augmentation_rate = augmentation_rate
        self.learning_rate = learning_rate

        self.strategy = tf.distribute.MirroredStrategy()
        print('目前使用gpu数量为: {}'.format(self.strategy.num_replicas_in_sync))
        if self.strategy.num_replicas_in_sync >= 8:
            print('[INFO]----------卡数上八!!!---------')

        if self.load_data_mode == 'hdf5':
            #   load_file_mode部分数据为part 便于测试 全部数据为all 其实也可以随便写 if part else all
            data_loader = Data_Loader_Hdf5(load_file_mode=self.load_file_mode, mask_size=self.mask_size,
                                           rewrite_hdf5=self.rewrite_hdf5, data_augmentation=self.data_augmentation,
                                           augmentation_rate=self.augmentation_rate, erase_rate=self.erase_rate)
            self.train_img, self.train_label = data_loader.load_train_data()
            self.val_img, self.val_label = data_loader.load_val_data()
        else:
            data_loader = Data_Loader_File(mask_size=self.mask_size, data_augmentation=False)
            self.train_img, self.train_label = data_loader.load_train_data(
                load_file_number=self.load_train_file_number)
            self.val_img, self.val_label = data_loader.load_val_data(
                load_file_number=self.load_val_file_number)

    def model_train(self):
        """
        可多卡训练
        :return:
        """
        with self.strategy.scope():
            # model = UNet_seg(filters=128, img_width=256, input_channel=3, num_class=151, num_con_unit=2)
            model = Deeplab_v3_plus(final_filters=151, num_middle=16, img_size=self.mask_size, input_channel=3,
                                    aspp_filters=512, final_activation=None)

            if self.learning_rate > 0:
                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )
            else:
                model.compile(
                    optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )

            if os.path.exists(self.checkpoint_save_path + '.index') and self.load_weights:
                print("[INFO] loading weights---------怕眼瞎看不见加长版--------")
                print("[INFO] loading weights---------怕眼瞎看不见加长版--------")
                print("[INFO] loading weights---------怕眼瞎看不见加长版--------")
                model.load_weights(self.checkpoint_save_path)
#   -------------------------------val_loss改成loss了 记得后头改过来
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_save_path,
                monitor='loss',
                save_weights_only=True,
                save_best_only=True,
                mode='auto',
                save_freq='epoch')

        history = model.fit(
            self.train_img, self.train_label, epochs=self.epochs, verbose=1, batch_size=self.batch_size,
            validation_data=(self.val_img, self.val_label), validation_freq=1, callbacks=[checkpoint_callback])

        model.summary()


def main():
    start_time = datetime.datetime.now()

    args = parseArgs()
    seg = seg_train(load_weights=args.load_weights, batch_size=args.batch_size, epochs=args.epochs,
                    load_data_mode=args.load_data_mode, mask_size=args.mask_size,
                    load_file_mode=args.load_file_mode, load_train_file_number=args.load_train_file_number,
                    load_val_file_number=args.load_val_file_number, rewrite_hdf5=args.rewrite_hdf5,
                    data_augmentation=args.data_augmentation, learning_rate=args.learning_rate)
    seg.model_train()

    print_cost_time(start_time)


if __name__ == '__main__':
    main()
