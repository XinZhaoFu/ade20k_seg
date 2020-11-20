import glob

import cv2
import numpy as np


part_train_img_file_path = './data/part_data/train/img/'
part_train_label_file_path = './data/part_data/train/label/'
part_val_img_file_path = './data/part_data/val/img/'
part_val_label_file_path = './data/part_data/val/label/'
part_test_img_file_path = './data/part_data/test/img/'
part_test_label_file_path = './data/part_data/test/label/'

train_img_file_list = glob.glob(part_train_img_file_path + '*.jpg')
train_label_file_list = glob.glob(part_train_label_file_path + '*.png')
val_img_file_list = glob.glob(part_val_img_file_path + '*.jpg')
val_label_file_list = glob.glob(part_val_label_file_path + '*.png')
test_img_file_list = glob.glob(part_test_img_file_path + '*.jpg')
test_label_file_list = glob.glob(part_test_label_file_path + '*.png')

print(len(train_img_file_list), len(train_label_file_list), len(val_img_file_list),
      len(val_label_file_list), len(test_img_file_list), len(test_label_file_list))


def get_numpy_data_list(img_file_list, label_file_list):
    dataset_img = np.empty((len(img_file_list), 512, 512, 3), dtype=np.float16)
    dataset_label = np.empty((len(label_file_list), 512, 512, 3), dtype=np.float16)
    index = 0
    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2.imread(img_file) / 255.
        label = cv2.imread(label_file)
        aug_img_list, aug_label_list = img_augmentation(img, label)
        for aug_img, aug_label in zip(aug_img_list, aug_label_list):
            aug_img = aug_img/255.
            if self.channel == 1:
                aug_img = np.reshape(aug_img, (self.img_width, self.img_width, 1))
            aug_label = aug_label/255.
            aug_label = np.reshape(aug_label, (self.img_width, self.img_width, 1))
            dataset_img[index, :, :, :] = aug_img
            for i in range(self.num_class):
                dataset_label[index, :, :, i] = (aug_label[:, :, 0] == i).astype(int)
            index += 1
    return dataset_img, dataset_label