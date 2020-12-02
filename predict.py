import glob

import cv2

from data_loader import Data_Loader
from model.unet import UNet_seg
import tensorflow as tf
import numpy as np

checkpoint_save_path = './checkpoint/unet_demo1.ckpt'
predict_save_path = './data/part_data/test/predict/'
test_label_file_path = './data/part_data/test/label/'
data_loader = Data_Loader(load_file_mode='part', mask_size=256, rewrite_hdf5=False)
test_img_list, _ = data_loader.load_test_data()

# 加载模型
model = UNet_seg()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.load_weights(checkpoint_save_path)

# 获取预测图的文件名
test_label_list = glob.glob(test_label_file_path + '*.png')
img_name_list = []
for img_file in test_label_list:
    img_name = (img_file.split('\\')[-1])
    img_name_list.append(img_name)

predict_list = model.predict(test_img_list, batch_size=12)

print(predict_list[0][100][100], predict_list[1][200][200], predict_list[2][100][100])

num_test_list = []
for _ in range(13):
    num_test_list.append(0)
print(num_test_list)

predict_img_list = []
for predict in predict_list:
    predict_img = np.ones(shape=(256, 256, 3), dtype=np.uint8)
    predict_img *= 128

    # predict_it = np.nditer(predict, flags=['multi_index', 'buffered'])
    # while not predict_it.finished:
    #     if predict_it[0]:
    #         predict_img[predict_it.multi_index[0]][predict_it.multi_index[1]][2] = predict_it.multi_index[2] * 10
    #     predict_it.iternext()

    for row in range(256):
        for col in range(256):
            for num_class in range(13):
                if predict[row][col][num_class]:
                    predict_img[row][col][2] = num_class * 10
                    num_test_list[num_class] += 1

    predict_img_list.append(predict_img)

for predict_img, img_name in zip(predict_img_list, img_name_list):
    cv2.imwrite(predict_save_path + img_name, predict_img)

print(num_test_list)