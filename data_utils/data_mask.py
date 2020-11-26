import glob
import numpy as np
import cv2
from utils import shuffle_file, write_hdf5
from math import ceil
from data_utils.data_augmentation import augmentation


def get_img_mask_hdf5(file_path, mask_size=512):
    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_file_list = glob.glob(img_path + '*.jpg')
    label_file_list = glob.glob(label_path + '*.png')
    assert len(img_file_list) == len(label_file_list)
    print(len(label_file_list))

    img_file_list, label_file_list = shuffle_file(img_file_list,label_file_list)

    num_class = 13
    num_file = 0

    img_array_hdf5 = np.empty(shape=(len(img_file_list), mask_size, mask_size, 3), dtype=np.float16)
    mask_array_hdf5 = np.zeros(shape=(len(label_file_list), mask_size, mask_size, num_class), dtype=np.uint8)

    img_list = []
    label_list = []

    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2.imread(img_file)
        img_list.append(img)
        label = cv2.imread(label_file)
        label_list.append(label)

    img_list, label_list = augmentation(img_list, label_list, mask_size=256)

    for img, label in zip(img_list, label_list):
        mask_temp = np.zeros(shape=(mask_size, mask_size, num_class))
        for w in range(mask_size):
            for l in range(mask_size):
                point_color = label[w][l]
                # class_point = int(point_color[2]/10) * 255 + point_color[1]
                class_point = ceil(point_color[2] / 10)
                if class_point > 12:
                    class_point = 12
                mask_temp[w][l][class_point] = 1

        img_array_hdf5[num_file, :, :, :] = img / 255.
        mask_array_hdf5[num_file, :, :, :] = mask_temp
        num_file += 1

        write_hdf5(img_array_hdf5, file_path + 'img.hdf5')
        write_hdf5(mask_array_hdf5, file_path + 'mask.hdf5')
