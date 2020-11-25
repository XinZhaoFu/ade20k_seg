import glob
import numpy as np
import cv2
from utils import shuffle_file, write_hdf5
from math import ceil


def get_img_mask_hdf5(file_path, mask_size=512):
    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_list = glob.glob(img_path + '*.jpg')
    label_list = glob.glob(label_path + '*.png')
    assert(len(img_list) == len(label_list))
    print(len(label_list))

    img_list, label_list = shuffle_file(img_list,label_list)

    num_class = 13
    num_file = 0

    img_array_hdf5 = np.empty(shape=(len(img_list), mask_size, mask_size, 3), dtype=np.float16)
    mask_array_hdf5 = np.zeros(shape=(len(label_list), mask_size, mask_size, num_class), dtype=np.uint8)

    for img_file, label_file in zip(img_list, label_list):

        print(num_file+1, img_file, label_file)

        img = cv2.imread(img_file)
        img = np.reshape(img, (mask_size, mask_size, 3))

        label = cv2.imread(label_file)
        label = np.reshape(label, (mask_size, mask_size, 3))

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