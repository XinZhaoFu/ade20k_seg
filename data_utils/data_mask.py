import glob
import numpy as np
import cv2
from utils import shuffle_file, write_hdf5
import gc
from data_utils.data_augmentation import augmentation, resize_img_label_list


def get_img_mask_list(file_number, file_path, mask_size=256, augmentation_mode=False,
                      augmentation_rate=1, erase_rate=0.1):
    """
    将图像和标签数据队列处理后返回
    图像格式为(size, size, 3)
    标签格式为(size, size, 1)
    标签总计151类(含背景)
    :param erase_rate:
    :param augmentation_rate:
    :param file_number:可以节取一部分数据
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

    # 截取部分文件
    if file_number > len(img_file_list):
        file_number = len(img_file_list)
    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)
    img_file_list, label_file_list = img_file_list[:file_number], label_file_list[:file_number]

    # 图片补足 以batchsize为64为假想
    num_supplement = 64 * ((len(img_file_list) // 64) + 1) - len(img_file_list)
    img_file_supplement_list = img_file_list[:num_supplement]
    label_file_supplement_list = label_file_list[:num_supplement]
    img_file_list.extend(img_file_supplement_list)
    label_file_list.extend(label_file_supplement_list)

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    img_array_hdf5 = np.empty(shape=(augmentation_rate * len(img_file_list), mask_size, mask_size, 3),
                              dtype=np.float16)
    mask_array_hdf5 = np.empty(shape=(augmentation_rate * len(label_file_list), mask_size, mask_size, 1),
                               dtype=np.uint8)

    file_index = 0
    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2.imread(img_file)
        label = cv2.imread(label_file)
        if augmentation_mode:
            for img, label in augmentation(img, label, mask_size=mask_size, erase_rate=erase_rate,
                                           augmentation_rate=augmentation_rate):
                img_array_hdf5[file_index, :, :, :] = img[:, :, :]
                mask_array_hdf5[file_index, :, :, 0] = label[:, :, 0]
                file_index += 1
        else:
            img = cv2.resize(img, (mask_size, mask_size))
            label = cv2.resize(label, (mask_size, mask_size))

            img_array_hdf5[file_index, :, :, :] = img[:, :, :] / np.float16(255.)
            mask_array_hdf5[file_index, :, :, 0] = label[:, :, 0]
            file_index += 1

        if file_index % 1000 == 0:
            print('已加载数目：\t' + str(file_index))

    return img_array_hdf5, mask_array_hdf5


def get_img_mask_hdf5(file_path, mask_size=256, augmentation_mode=False, augmentation_rate=1, erase_rate=0.1):
    """
    将图像和标签存为hdf5文件
    推荐真正的猛男在服务器上使用这种方式一次性把所有文件都读取到内存里
    节省io时间
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
    img_file_list = glob.glob(img_path + '*.jpg')
    label_file_list = glob.glob(label_path + '*.png')
    assert len(img_file_list) == len(label_file_list)

    # 图片补足 以batchsize为64为假想
    num_supplement = 64 * ((len(img_file_list) // 64) + 1) - len(img_file_list)
    img_file_supplement_list = img_file_list[:num_supplement]
    label_file_supplement_list = label_file_list[:num_supplement]
    img_file_list.extend(img_file_supplement_list)
    label_file_list.extend(label_file_supplement_list)

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    if augmentation_mode is True:
        img_array_hdf5 = np.empty(shape=(augmentation_rate*len(img_file_list), mask_size, mask_size, 3),
                                  dtype=np.float16)
        mask_array_hdf5 = np.empty(shape=(augmentation_rate*len(label_file_list), mask_size, mask_size, 1),
                                   dtype=np.uint8)
    else:
        img_array_hdf5 = np.empty(shape=(len(img_file_list), mask_size, mask_size, 3), dtype=np.float16)
        mask_array_hdf5 = np.empty(shape=(len(label_file_list), mask_size, mask_size, 1), dtype=np.uint8)

    file_index = 0
    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2.imread(img_file)
        label = cv2.imread(label_file)
        if augmentation_mode is True:
            print('进行数据扩增 扩增倍数为：\t' + str(augmentation_rate) + '遮盖比例为：\t' + str(erase_rate))
            for img, label in augmentation(img, label, mask_size=mask_size, erase_rate=erase_rate,
                                           augmentation_rate=augmentation_rate):
                img_array_hdf5[file_index, :, :, :] = img[:, :, :] / np.float16(255.)
                mask_array_hdf5[file_index, :, :, 0] = label[:, :, 0]
                file_index += 1
        else:
            img = cv2.resize(img, (mask_size, mask_size))
            label = cv2.resize(label, (mask_size, mask_size))
            img_array_hdf5[file_index, :, :, :] = img[:, :, :] / np.float16(255.)
            mask_array_hdf5[file_index, :, :, 0] = label[:, :, 0]
            file_index += 1

        if file_index % 1000 == 0:
            print('已加载数目：\t' + str(file_index))

    write_hdf5(img_array_hdf5, file_path + 'img.hdf5')
    write_hdf5(mask_array_hdf5, file_path + 'mask.hdf5')

    del img_array_hdf5
    del mask_array_hdf5
    gc.collect()
