import glob
import numpy as np
import cv2
from utils import shuffle_file, write_hdf5
from math import ceil
from data_utils.data_augmentation import augmentation, resize_img_label_list


def get_img_mask_hdf5(file_path, mask_size=512, augmentation_mode=0):
    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_file_list = glob.glob(img_path + '*.jpg')
    label_file_list = glob.glob(label_path + '*.png')
    assert len(img_file_list) == len(label_file_list)

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    num_class = 13
    num_file = 0
    #   test
    num_test_list = []
    for _ in range(13):
        num_test_list.append(0)

    img_array_hdf5 = np.empty(shape=(len(img_file_list), mask_size, mask_size, 3), dtype=np.float16)
    mask_array_hdf5 = np.empty(shape=(len(label_file_list), mask_size, mask_size, num_class), dtype=np.uint8)

    img_list = []
    label_list = []

    """
    这里实在是太吃内存了 在数据增强前不得不做一个截流 每次只通过一定数量的文件
    因为把resize放在数据增强这个函数里了
    经过resize后的数据基本就能受得住了
    """
    data_loader_batch_size = 100
    count_temp_for_memory = 1
    count_list_for_memory = 1
    img_list_temp = []
    label_list_temp = []

    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2.imread(img_file)
        label = cv2.imread(label_file)
        img_list_temp.append(img)
        label_list_temp.append(label)

        count_temp_for_memory += 1
        count_list_for_memory += 1

        if count_temp_for_memory % data_loader_batch_size == 0 or count_list_for_memory == len(img_file_list):
            print('已加载' + str(int(count_list_for_memory/data_loader_batch_size)) + '批次\t共计：' +
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

    for img, label in zip(img_list, label_list):
        nd_label_temp = np.empty(shape=(mask_size, mask_size))
        nd_label_temp[:, :] = label[:, :, 2]
        mask_temp = np.zeros(shape=(mask_size, mask_size, num_class), dtype=np.int)

        for row in range(mask_size):
            for col in range(mask_size):
                if nd_label_temp[row][col]:
                    point_class = int(nd_label_temp[row][col] / 10)
                    if point_class > 12:
                        point_class = 12
                    if point_class < 0:
                        point_class = 0
                    mask_temp[row, col, point_class] = 1
                    # test
                    num_test_list[point_class] += 1

        img_array_hdf5[num_file, :, :, :] = img / 255.
        mask_array_hdf5[num_file, :, :, :] = mask_temp
        num_file += 1
        if num_file % 100 == 0:
            print('已完成' + str(int(num_file)) + '份独热码的编码工作')

    write_hdf5(img_array_hdf5, file_path + 'img.hdf5')
    write_hdf5(mask_array_hdf5, file_path + 'mask.hdf5')

    print('逐点统计信息：')
    print(num_test_list)