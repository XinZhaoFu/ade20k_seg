from data_utils.data_mask import get_img_mask_list


class Data_Loader_File:
    def __init__(self, mask_size, data_augmentation=False, batch_size=16):
        self.batch_size = batch_size
        self.mask_size = mask_size
        self.data_augmentation = data_augmentation

        self.train_file_path = './data/train/'
        self.val_file_path = './data/val/'
        self.test_file_path = './data/test/'

    def load_train_data(self, load_file_number=1000):
        print('正在载入训练集')
        train_dataset = get_img_mask_list(file_number=load_file_number,
                                          file_path=self.train_file_path,
                                          mask_size=self.mask_size,
                                          augmentation_mode=self.data_augmentation,
                                          batch_size=self.batch_size)
        return train_dataset

    def load_val_data(self, load_file_number=200):
        print('正在载入验证集')
        val_dataset = get_img_mask_list(file_number=load_file_number,
                                        file_path=self.val_file_path,
                                        mask_size=self.mask_size,
                                        batch_size=self.batch_size)
        return val_dataset

    def load_test_data(self, load_file_number=200):
        print('正在载入测试集')
        test_dataset = get_img_mask_list(file_number=load_file_number,
                                         file_path=self.test_file_path,
                                         mask_size=self.mask_size,
                                         batch_size=self.batch_size)

        return test_dataset
