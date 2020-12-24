import glob
import numpy as np
import cv2
import tensorflow as tf
from utils import shuffle_file, write_hdf5
import gc
from data_utils.data_augmentation import augmentation, resize_img_label_list


def get_img_mask_hdf5(file_path, mask_size=256, augmentation_mode=False):
    """
    将图像和标签存为hdf5文件
    图像格式为(size, size, 3)
    标签格式为(size, size, 1)
    标签总计151类(含背景)
    :param file_path:
    :param mask_size:
    :param augmentation_mode:
    :return:
    """
    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_file_list = glob.glob(img_path + '*.jpg')
    label_file_list = glob.glob(label_path + '*.png')
    assert len(img_file_list) == len(label_file_list)

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    img_list = []
    label_list = []
    """
    这里实在是太吃内存了 在数据增强前不得不做一个截流 每次只通过一定数量的文件
    因为把resize放在数据增强这个函数里了
    缩小后的数据基本就能受得住了
    """
    data_loader_batch_size = 500
    count_list_for_memory = 1
    img_list_temp = []
    label_list_temp = []

    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2.imread(img_file)
        label = cv2.imread(label_file)
        img_list_temp.append(img)
        label_list_temp.append(label)

        count_list_for_memory += 1
        if count_list_for_memory % data_loader_batch_size == 0 or count_list_for_memory == len(img_file_list):
            print('已加载' + str(int(count_list_for_memory / data_loader_batch_size)) + '批次\t共计：' +
                  str(int(count_list_for_memory)) + '个文件')
            if augmentation_mode:
                img_list_temp, label_list_temp = augmentation(img_list_temp, label_list_temp,
                                                              mask_size=256, erase_rate=0.2)
            else:
                img_list_temp, label_list_temp = resize_img_label_list(img_list_temp, label_list_temp, mask_size=256)
            img_list.extend(img_list_temp)
            label_list.extend(label_list_temp)
            img_list_temp.clear()
            label_list_temp.clear()

    del img_list_temp
    del label_list_temp
    gc.collect()

    img_array_hdf5 = np.empty(shape=(len(img_file_list), mask_size, mask_size, 3), dtype=np.uint8)
    mask_array_hdf5 = np.empty(shape=(len(label_file_list), mask_size, mask_size, 1), dtype=np.uint8)

    num_file = 0
    for img, label in zip(img_list, label_list):
        img_array_hdf5[num_file, :, :, :] = img
        mask_array_hdf5[num_file, :, :, 0] = label[:, :, 0]
        num_file += 1

    del img_list
    del label_list
    gc.collect()

    write_hdf5(img_array_hdf5, file_path + 'img.hdf5')
    write_hdf5(mask_array_hdf5, file_path + 'mask.hdf5')

    del img_array_hdf5
    del mask_array_hdf5
    gc.collect()


def get_img_mask_list(file_number, file_path, mask_size=256, augmentation_mode=False):
    """
    将图像和标签数据队列处理后返回
    图像格式为(size, size, 3)
    标签格式为(size, size, 1)
    标签总计151类(含背景)
    :param file_number:
    :param file_path:
    :param mask_size:
    :param augmentation_mode:
    :return:
    """
    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_file_list = glob.glob(img_path + '*.jpg')
    label_file_list = glob.glob(label_path + '*.png')

    assert len(img_file_list) == len(label_file_list)

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)
    img_file_list = img_file_list[:file_number]
    label_file_list = label_file_list[:file_number]

    img_list = []
    label_list = []
    """
    同上 吃内存
    """
    data_loader_batch_size = 500
    count_list_for_memory = 0
    img_list_temp = []
    label_list_temp = []

    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2.imread(img_file)
        label = cv2.imread(label_file)
        img_list_temp.append(img)
        label_list_temp.append(label)

        count_list_for_memory += 1
        if count_list_for_memory % data_loader_batch_size == 0 or count_list_for_memory == len(img_file_list):
            print('已加载' + str(count_list_for_memory // data_loader_batch_size) + '批次\t共计：' +
                  str(int(count_list_for_memory)) + '个文件')
            if augmentation_mode:
                img_list_temp, label_list_temp = augmentation(img_list_temp, label_list_temp,
                                                              mask_size=mask_size, erase_rate=0.2)
            else:
                img_list_temp, label_list_temp = resize_img_label_list(img_list_temp, label_list_temp,
                                                                       mask_size=mask_size)
            img_list.extend(img_list_temp)
            label_list.extend(label_list_temp)
            img_list_temp.clear()
            label_list_temp.clear()

    del img_list_temp
    del label_list_temp
    gc.collect()

    img_ndarray = np.empty(shape=(len(img_file_list), mask_size, mask_size, 3), dtype=np.float16)
    label_ndarray = np.empty(shape=(len(label_file_list), mask_size, mask_size, 1), dtype=np.uint8)

    num_file = 0
    for img, label in zip(img_list, label_list):
        img_ndarray[num_file, :, :, :] = img / 255.
        label_ndarray[num_file, :, :, 0] = label[:, :, 0]
        num_file += 1

    del img_list
    del label_list
    gc.collect()

    return img_ndarray, label_ndarray
