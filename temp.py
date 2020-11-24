import glob
import os

import cv2
import numpy as np
import scipy.io as scio
import pandas as pd
import scipy


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

# train_img_file_list = glob.glob(part_train_img_file_path + '*.jpg')
# train_label_file_list = glob.glob(part_train_label_file_path + '*.png')
# val_img_file_list = glob.glob(part_val_img_file_path + '*.jpg')
# val_label_file_list = glob.glob(part_val_label_file_path + '*.png')
# test_img_file_list = glob.glob(part_test_img_file_path + '*.jpg')
# test_label_file_list = glob.glob(part_test_label_file_path + '*.png')
#
#
# def resize_img_file(img_file_path):
#     img = cv2.imread(img_file_path)
#     img = cv2.resize(img, dsize=(512, 512))
#     cv2.imwrite(img_file_path, img)
#
#
# file_list = train_img_file_list + train_label_file_list + val_img_file_list + \
#             val_label_file_list + test_img_file_list + test_label_file_list
#
# for file in file_list:
#     resize_img_file(file)

img = cv2.imread('./data/test/label/' + 'ADE_train_00002141_seg.png')
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
