from glob import glob
import numpy as np
import cv2
from utils import shuffle_file, write_hdf5
from gc import collect
import tensorflow as tf
from data_utils.data_augmentation import augmentation

tf_decode_jpeg = tf.image.decode_jpeg
tf_resize = tf.image.resize
tf_decode_png = tf.image.decode_png
tf_read_file = tf.io.read_file
tf_cast = tf.cast
tf_uint8 = tf.uint8


def get_img_mask_list(file_number, file_path, mask_size=256, augmentation_mode=False,
                      augmentation_rate=1, erase_rate=0.1, batch_size=16):
    """
    将图像和标签数据队列处理后以tensor返回
    图像格式为(size, size, 3)
    标签格式为(size, size, 1)
    标签总计151类(含背景)
    :param batch_size:
    :param erase_rate:
    :param augmentation_rate:
    :param file_number:可以节取一部分数据
    :param file_path:
    :param mask_size:
    :param augmentation_mode:
    :return:
    """
    autotune = tf.data.experimental.AUTOTUNE

    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_file_path_list = glob(img_path + '*.jpg')
    label_file_path_list = glob(label_path + '*.png')

    assert len(img_file_path_list) == len(label_file_path_list)

    # 截取部分文件
    if file_number > len(img_file_path_list):
        file_number = len(img_file_path_list)
    img_file_path_list, label_file_path_list = shuffle_file(img_file_path_list, label_file_path_list)
    img_file_path_list, label_file_path_list = img_file_path_list[:file_number], label_file_path_list[:file_number]

    # # 图片补足 以batchsize为64为假想
    # num_supplement = 64 * ((len(img_file_path_list) // 64) + 1) - len(img_file_path_list)
    # img_file_supplement_list = img_file_path_list[:num_supplement]
    # label_file_supplement_list = label_file_path_list[:num_supplement]
    # img_file_path_list.extend(img_file_supplement_list)
    # label_file_path_list.extend(label_file_supplement_list)

    img_file_path_list, label_file_path_list = shuffle_file(img_file_path_list, label_file_path_list)

    img_file_path_ds = tf.data.Dataset.from_tensor_slices(img_file_path_list)
    image_ds = img_file_path_ds.map(load_and_preprocess_image, num_parallel_calls=autotune)
    label_file_path_ds = tf.data.Dataset.from_tensor_slices(label_file_path_list)
    label_ds = label_file_path_ds.map(load_and_preprocess_label, num_parallel_calls=autotune)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    image_label_ds = image_label_ds.shuffle(buffer_size=batch_size * 8)
    # image_label_ds = image_label_ds.repeat()
    image_label_ds = image_label_ds.batch(batch_size)
    image_label_ds = image_label_ds.prefetch(buffer_size=autotune)
    image_label_ds = image_label_ds.cache(filename='data/cache')
    return image_label_ds


def get_img_mask_hdf5(file_path, mask_size=256, augmentation_mode=False, augmentation_rate=1, erase_rate=0.1):
    """
    将图像和标签存为hdf5文件
    真正的猛男敢于在服务器上使用这种方式一次性把所有文件都读取到内存里
    节省io时间 同时锻炼抗击打能力
    图像格式为(size, size, 3)
    标签格式为(size, size, 1)
    标签总计151类(含背景)
    :param erase_rate:
    :param augmentation_rate:
    :param file_path:
    :param mask_size:
    :param augmentation_mode:
    :return:
    """
    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_file_list = glob(img_path + '*.jpg')
    label_file_list = glob(label_path + '*.png')
    assert len(img_file_list) == len(label_file_list)

    # 图片补足 以batchsize为64为假想
    num_supplement = 64 * ((len(img_file_list) // 64) + 1) - len(img_file_list)
    img_file_supplement_list = img_file_list[:num_supplement]
    label_file_supplement_list = label_file_list[:num_supplement]
    img_file_list.extend(img_file_supplement_list)
    label_file_list.extend(label_file_supplement_list)

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    if augmentation_mode is True:
        img_array_hdf5 = np.empty(shape=(augmentation_rate * len(img_file_list), mask_size, mask_size, 3),
                                  dtype=np.float16)
        mask_array_hdf5 = np.empty(shape=(augmentation_rate * len(label_file_list), mask_size, mask_size, 1),
                                   dtype=np.uint8)
    else:
        img_array_hdf5 = np.empty(shape=(len(img_file_list), mask_size, mask_size, 3), dtype=np.float16)
        mask_array_hdf5 = np.empty(shape=(len(label_file_list), mask_size, mask_size, 1), dtype=np.uint8)

    file_index = 0
    cv2imread = cv2.imread
    cv2resize = cv2.resize
    npfloat16 = np.float16
    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2imread(img_file)
        label = cv2imread(label_file)

        img_name = (img_file.split('\\')[-1]).split('.')[0]
        label_name = (label_file.split('\\')[-1]).split('.')[0]
        assert img_name == label_name

        if augmentation_mode is True:
            for img, label in augmentation(img, label, mask_size=mask_size, erase_rate=erase_rate,
                                           augmentation_rate=augmentation_rate):
                img_array_hdf5[file_index, :, :, :] = img[:, :, :] / npfloat16(255.)
                mask_array_hdf5[file_index, :, :, 0] = label[:, :, 0]
                file_index += 1
        else:
            img = cv2resize(img, (mask_size, mask_size))
            label = cv2resize(label, (mask_size, mask_size))
            img_array_hdf5[file_index, :, :, :] = img[:, :, :] / np.float16(255.)
            mask_array_hdf5[file_index, :, :, 0] = label[:, :, 0]
            file_index += 1

        if file_index % 1000 == 0:
            print('已加载数目：\t' + str(file_index))

    write_hdf5(img_array_hdf5, file_path + 'img.hdf5')
    write_hdf5(mask_array_hdf5, file_path + 'mask.hdf5')

    del img_array_hdf5
    del mask_array_hdf5
    collect()


def load_and_preprocess_image(path):
    image = tf_read_file(path)
    image = tf_decode_jpeg(image, channels=3)
    image = tf_resize(image, [256, 256])
    image /= 255.0
    return image


def load_and_preprocess_label(path):
    image = tf_read_file(path)
    image = tf_decode_png(image)
    image = tf_resize(image, [256, 256])
    image = tf_cast(image, dtype=tf_uint8)

    return image
