import datetime
from model.deeplab_v3_plus import Deeplab_v3_plus
import tensorflow as tf
import os
from data_utils.data_loader_hdf5 import Data_Loader_Hdf5
from data_utils.data_loader_file import Data_Loader_File
from utils import print_cost_time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class seg_train:
    def __init__(self):
        self.load_weights = False
        # checkpoint_save_path = './checkpoint/unet_demo1.ckpt'
        self.checkpoint_save_path = './checkpoint/deeplabv3plus_demo1.ckpt'
        self.batch_size = 8
        self.epochs = 100
        self.load_data_mode = 'file'
        self.mask_size = 256

        if self.load_data_mode == 'hdf5':
            #   load_file_mode部分数据为part 便于测试 全部数据为all 其实也可以随便写 if part else all
            data_loader = Data_Loader_Hdf5(load_file_mode='part', mask_size=self.mask_size,
                                           rewrite_hdf5=False, data_augmentation=False)

            self.train_img, self.train_label = data_loader.load_train_data()
            self.val_img, self.val_label = data_loader.load_val_data()
        else:
            data_loader = Data_Loader_File(mask_size=self.mask_size, data_augmentation=False)
            self.train_img, self.train_label = data_loader.load_train_data(load_file_number=1000)
            self.val_img, self.val_label = data_loader.load_val_data(load_file_number=100)

        self.train_img = self.train_img / 255.
        self.val_img = self.val_img / 255.

        print(self.train_img.shape, self.train_label.shape, self.val_img.shape, self.val_label.shape)

    def model_train(self):
        # model = UNet_seg(filters=128, img_width=256, input_channel=3, num_class=13, num_con_unit=2)
        model = Deeplab_v3_plus(final_filters=151, num_middle=8, img_size=256, input_channel=3)

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        if os.path.exists(self.checkpoint_save_path+'.index') and self.load_weights:
            print("[INFO] loading weights")
            model.load_weights(self.checkpoint_save_path)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_save_path,
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            mode='auto',
            save_freq='epoch'
        )

        history = model.fit(
            self.train_img, self.train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
            validation_data=(self.val_img, self.val_label), validation_freq=1, callbacks=[checkpoint_callback]
        )

        model.summary()


def main():
    start_time = datetime.datetime.now()

    seg = seg_train()
    seg.model_train()

    print_cost_time(start_time)


if __name__ == '__main__':
    main()

