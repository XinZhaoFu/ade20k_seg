from utils import load_hdf5
from data_utils.data_mask import get_img_mask_hdf5, get_img_mask_onehot_hdf5


class Data_Loader:
    def __init__(self, load_file_mode, mask_size, rewrite_temp_hdf5=False, rewrite_onehot_hdf5=False):
        self.load_file_mode = load_file_mode
        self.mask_size = mask_size
        self.rewrite_temp_hdf5 = rewrite_temp_hdf5
        self.rewrite_onehot_dhf5 = rewrite_onehot_hdf5

        if load_file_mode == 'part':
            self.train_file_path = './data/part_data/train/'
            self.val_file_path = './data/part_data/val/'
            self.test_file_path = './data/part_data/test/'
        else:
            self.train_file_path = './data/train/'
            self.val_file_path = './data/val/'
            self.test_file_path = './data/test/'

        if rewrite_onehot_hdf5:
            if rewrite_temp_hdf5:
                self.rewrite_temp_hdf5_file()
            self.rewrite_onehot_hdf5_file()

    def rewrite_onehot_hdf5_file(self):
        print('正在重写独热码hdf5文件---------')
        print('正在重写独热码train_hdf5文件')
        get_img_mask_onehot_hdf5(file_path=self.train_file_path, num_class=13)
        print('正在重写独热码val_hdf5文件')
        get_img_mask_onehot_hdf5(file_path=self.val_file_path, num_class=13)
        print('正在重写独热码test_hdf5文件')
        get_img_mask_onehot_hdf5(file_path=self.test_file_path, num_class=13)
        print('独热码重写完了')

    def rewrite_temp_hdf5_file(self):
        """
        重写中间态
        :return:
        """
        print('正在重写中间态hdf5文件---------')
        print('正在重写中间态train_temp_hdf5文件')
        get_img_mask_hdf5(file_path=self.train_file_path, mask_size=self.mask_size, augmentation_mode=1)
        print('正在重写中间态val_temp_hdf5文件')
        get_img_mask_hdf5(file_path=self.val_file_path, mask_size=self.mask_size)
        print('正在重写中间态test_temp_hdf5文件')
        get_img_mask_hdf5(file_path=self.test_file_path, mask_size=self.mask_size)
        print('中间态重写完了')

    def load_train_data(self):
        print('正在载入训练集')
        train_img_dataset = load_hdf5(self.train_file_path + 'img.hdf5')
        train_mask_dataset = load_hdf5(self.train_file_path + 'mask.hdf5')
        return train_img_dataset, train_mask_dataset

    def load_val_data(self):
        print('正在载入验证集')
        val_img_dataset = load_hdf5(self.val_file_path + 'img.hdf5')
        val_mask_dataset = load_hdf5(self.val_file_path + 'mask.hdf5')
        return val_img_dataset, val_mask_dataset

    def load_test_data(self):
        print('正在载入测试集')
        test_img_dataset = load_hdf5(self.test_file_path + 'img.hdf5')
        test_mask_dataset = load_hdf5(self.test_file_path + 'mask.hdf5')
        return test_img_dataset, test_mask_dataset
