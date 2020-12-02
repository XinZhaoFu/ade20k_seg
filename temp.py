import glob
import os
from math import ceil

from data_loader import Data_Loader
from data_utils.data_augmentation import img_crop
from random import randint
import cv2
import numpy as np
import scipy.io as scio
import pandas as pd
import scipy
import tensorflow as tf
from random import choice
from data_utils.data_augmentation import augmentation


np.set_printoptions(threshold=np.inf)
# ori_img_file_path = './data/ori-img/'
# ori_label_file_path = './data/ori-label/'
#
# ori_img_file_list = glob.glob(ori_img_file_path + '*.jpg')
# ori_label_file_list = glob.glob(ori_label_file_path + '*.png')
#
# print(len(ori_img_file_list), len(ori_label_file_list))

part_train_img_file_path = './data/part_data/train/img/'
part_train_label_file_path = './data/part_data/train/label/'
part_val_img_file_path = './data/part_data/val/img/'
part_val_label_file_path = './data/part_data/val/label/'
part_test_img_file_path = './data/part_data/test/img/'
part_test_label_file_path = './data/part_data/test/label/'

data_loader = Data_Loader(load_file_mode='all', mask_size=256, rewrite_hdf5=False)
train_img_list, train_label_list = data_loader.load_train_data()

print(train_label_list[0][100][100], train_label_list[1][100][100], train_label_list[2][100][100])

# for _ in range(10):
#     print(choice([0, 1]))
# #
# mask_size = 256
# num_class = 13

# img = cv2.imread('./data/part_data/train/img/ADE_train_00000065.jpg')
# label = cv2.imread('./data/part_data/train/label/ADE_train_00000065_seg.png')
#
# label = cv2.resize(label, dsize=(mask_size, mask_size))
# nd_label_temp = np.empty(shape=(mask_size, mask_size))
# nd_label_temp[:, :] = label[:, :, 2]
# mask_temp = np.zeros(shape=(mask_size, mask_size, num_class), dtype=np.uint8)
#
# label_it = np.nditer(nd_label_temp, flags=['multi_index', 'buffered'])
# while not label_it.finished:
#     class_point = ceil(label_it[0] / 10.)
#     print(class_point)
#     if class_point > 12:
#         class_point = 12
#     mask_temp[label_it.multi_index[0], label_it.multi_index[1], class_point] = 1
#     # print(mask_temp[label_it.multi_index[0]][label_it.multi_index[1]][class_point])
#     label_it.iternext()


# for row in range(mask_size):
#     for col in range(mask_size):
#         if nd_label_temp[row][col]:
#             num_class = int(nd_label_temp[row][col] / 10)
#             # print(row, col, num_class)
#             mask_temp[row, col, num_class] = 1


# print(mask_temp)

# temp_2d_array = np.zeros(shape=(100, 100), dtype=np.int)
#
# label_it = np.nditer(temp_2d_array, op_flags=['readwrite'])
# while not label_it.finished:
#     label_it[0] = randint(0, 12)
#     label_it.iternext()
# print(temp_2d_array)
# # temp_2d_array = [[0, 1, 1], [1, 2, 1]]
# num_class = 13
# temp_2d_array = tf.one_hot(temp_2d_array, num_class)
# print(temp_2d_array)

# img = cv2.resize(img, (256, 256))
# nd_img = np.empty(shape=(256, 256, 3), dtype=np.uint8)
# nd_img[:, :, :] = img
#
# a = np.arange(6).reshape(2, 3)
#
# for point in np.nditer(nd_img, flags=['external_loop', 'buffered'], order='K'):
#     print(point)


# a = np.arange(6).reshape(2, 3)
# b = np.zeros(shape=(2, 3))
#
# it = np.nditer(a, flags=['multi_index'])
# while not it.finished:
#     print("%d <%s>" % (it[0], it.multi_index))
#     b[it.multi_index[0]][it.multi_index[1]] = it[0]
#     it.iternext()
#
# print(b)

# img_list, label_list = [], []
# for _ in range(100):
#     img_list.append(img)
#     label_list.append(label)
#
# aug_img_list, aug_label_list = augmentation(img_list, label_list, mask_size=256)


#
# crop_img, cop_label = img_crop(img, label, mask_size=256)
# cv2.imwrite('./data/test0.jpg', crop_img)
# cv2.imwrite('./data/test0.png', cop_label)


# def stand_color(label):
#     for w in range(512):
#         for l in range(512):
#             point_color = label[w][l]
#             if point_color[2] % 10 != 0:
#                 label[w][l][2] = 0
#     return label

#
# label = cv2.imread('./data/part_data/train/label/ADE_train_00000120_seg.png')
# cv2.imwrite('./data/test0.png', label)
# label = stand_color(label)
# cv2.imwrite('./data/test1.png', label)



# def get_colors_dict(label_path):
#     label_list = glob.glob(label_path + '*.png')
#     print(len(label_list))
#     num_file = 1
#     account_list = np.zeros(shape=(256, 1), dtype=np.int)
#
#     for label_file in label_list:
#         label = cv2.imread(label_file)
#         for w in range(512):
#             for l in range(512):
#                 point_color = label[w][l]
#                 account_list[point_color[2]] += 1
#         print('已完成数目:' + str(num_file))
#         num_file += 1
#
#     for i in range(256):
#         print(i, account_list[i])
#
#
# get_colors_dict(part_train_label_file_path)

# train_img_file_list = glob.glob(part_train_img_file_path + '*.jpg')
# train_label_file_list = glob.glob(part_train_label_file_path + '*.png')
# val_img_file_list = glob.glob(part_val_img_file_path + '*.jpg')
# val_label_file_list = glob.glob(part_val_label_file_path + '*.png')
# test_img_file_list = glob.glob(part_test_img_file_path + '*.jpg')
# test_label_file_list = glob.glob(part_test_label_file_path + '*.png')


# def resize_img_file(img_file_path):
#     img = cv2.imread(img_file_path)
#     img = cv2.resize(img, dsize=(512, 512))
#     cv2.imwrite(img_file_path, img)


# file_list = train_img_file_list + train_label_file_list + val_img_file_list + \
#             val_label_file_list + test_img_file_list + test_label_file_list

# for file in file_list:
#     resize_img_file(file)

# img = cv2.imread('./data/test/label/' + 'ADE_train_00002141_seg.png')
# img_name = (file_path.split('\\')[-1]).split('.')[0]
# r = str(img[300][300]).split(' ')[0]
# g = str(img[300][300]).split(' ')[1]
# print(type(img))
# str_color = (str(img[300][300]).split('[')[-1]).split(']')[0]
# print(str_color)
# print(str_color.split())
# r = str_color.split()[0]
# g = str_color.split()[1]
# b = str_color.split()[2]
# color_str = r + ' ' + g + ' ' + b + ' '
# print(r, g, b)
# print(color_str)
# print(type(img[300][300]))
# print(type((img[300][300][0], img[300][300][1], img[300][300][2])))


# cv2.imwrite('./data/test0.png', img)
# r_img = (img[:, :, 0] / 10) + img[:, :, 1]
# # r_img = img[:, :, 2]
# cv2.imwrite('./data/test1.png', r_img)


# matPath = './data/'
#
# outPath = './data/'
#
# for i in os.listdir(matPath):
#     inputFile = os.path.join(matPath, i)
#
#     outputFile = os.path.join(outPath, os.path.split(i)[1][:-4] + '.csv')
#
#     features_struct = scipy.io.loadmat(inputFile)
#
#     data = list(features_struct.values())[-1]
#
#     dfdata = pd.DataFrame(data)
#
#     dfdata.to_csv(outputFile, index=False)


# mat_path = './data/color150.mat'
# mat_data = scio.loadmat(mat_path)
# print(mat_data.keys())
# # print(type(mat_data))
# colors = mat_data['colors']
# print(colors)
# print(data.shape)
# print(data[0][0])

# mat_path = './data/index_ade20k.mat'
# mat_data = scio.loadmat(mat_path)
# print(mat_data.keys())
#
# index = mat_data['index']
# print(index)
