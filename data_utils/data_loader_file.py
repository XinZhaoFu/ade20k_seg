from glob import glob
from data_utils.data_utils import check_img_label_list
import tensorflow as tf

tf_decode_jpeg = tf.image.decode_jpeg
tf_resize = tf.image.resize
tf_decode_png = tf.image.decode_png
tf_read_file = tf.io.read_file
tf_cast = tf.cast
tf_uint8 = tf.uint8


class Data_Loader_File:
    def __init__(self,
                 img_size=512,
                 mask_size=64,
                 data_augmentation=False,
                 batch_size=16):
        self.img_size = img_size
        self.batch_size = batch_size
        self.mask_size = mask_size
        self.data_augmentation = data_augmentation

        self.train_file_path = './data/train/'
        self.val_file_path = './data/val/'
        self.test_file_path = './data/test/'

    def load_train_data(self, load_file_number=1000):
        print('正在载入训练集')
        train_dataset = self.get_img_mask_list(file_number=load_file_number,
                                          file_path=self.train_file_path,
                                               data_augmentation=self.data_augmentation)
        return train_dataset

    def load_val_data(self, load_file_number=200):
        print('正在载入验证集')
        val_dataset = self.get_img_mask_list(file_number=load_file_number,
                                        file_path=self.val_file_path)
        return val_dataset

    def load_test_data(self, load_file_number=200):
        print('正在载入测试集')
        test_dataset = self.get_img_mask_list(file_number=load_file_number,
                                         file_path=self.test_file_path)

        return test_dataset

    def get_img_mask_list(self, file_path, file_number=0, data_augmentation=False):
        """
        将图像和标签数据队列处理后以tensor返回
        图像格式为(size, size, 3)
        标签格式为(size, size, 1)
        标签总计151类(含背景)
        :param data_augmentation:
        :param file_number:可以节取一部分数据
        :param file_path:
        :return:
        """
        autotune = tf.data.experimental.AUTOTUNE

        img_path = file_path + 'img/'
        label_path = file_path + 'label/'

        if data_augmentation:
            print('调用数据增强后的文件')
            img_path = file_path + 'aug_img/'
            label_path = file_path + 'aug_label/'

        img_file_path_list = glob(img_path + '*.jpg')
        label_file_path_list = glob(label_path + '*.png')

        assert len(img_file_path_list) == len(label_file_path_list)

        # 截取部分文件
        if file_number > 0:
            print('截取部分文件 其数量为：\t' + str(len(img_file_path_list)))
            if file_number > len(img_file_path_list):
                file_number = len(img_file_path_list)
            img_file_path_list, label_file_path_list = img_file_path_list[:file_number], label_file_path_list[:file_number]
        else:
            print('不截取文件 其数量为：\t' + str(len(img_file_path_list)))

        # 文件对应检查
        check_img_label_list(img_file_path_list, label_file_path_list)

        img_file_path_ds = tf.data.Dataset.from_tensor_slices(img_file_path_list)
        image_ds = img_file_path_ds.map(self.load_and_preprocess_image, num_parallel_calls=autotune)
        label_file_path_ds = tf.data.Dataset.from_tensor_slices(label_file_path_list)
        label_ds = label_file_path_ds.map(self.load_and_preprocess_label, num_parallel_calls=autotune)
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

        image_label_ds = image_label_ds.cache(filename='data/cache/')
        image_label_ds = image_label_ds.shuffle(buffer_size=self.batch_size * 8)
        # image_label_ds = image_label_ds.repeat()
        image_label_ds = image_label_ds.batch(self.batch_size)
        image_label_ds = image_label_ds.prefetch(buffer_size=autotune)

        return image_label_ds

    def load_and_preprocess_image(self, path):
        image = tf_read_file(path)
        image = tf_decode_jpeg(image, channels=3)
        image = tf_resize(image, [self.img_size, self.img_size])
        image /= 255.0
        return image

    def load_and_preprocess_label(self, path):
        image = tf_read_file(path)
        image = tf_decode_png(image, channels=1)
        image = tf_resize(image, [self.mask_size, self.mask_size])
        image = tf_cast(image, dtype=tf_uint8)

        return image


