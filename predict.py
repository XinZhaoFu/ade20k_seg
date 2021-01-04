import glob
import cv2
from utils import onehot_to_class
from data_utils.data_loader_hdf5 import Data_Loader_Hdf5
from model.unet import UNet_seg
from model.deeplab_v3_plus import Deeplab_v3_plus
import tensorflow as tf
import numpy as np

checkpoint_save_path = './checkpoint/deeplabv3plus_demo1.ckpt'
predict_save_path = './data/part_data/test/predict/'
test_label_file_path = './data/part_data/test/label/'

data_loader = Data_Loader_Hdf5(load_file_mode='part', mask_size=256)
test_img_list, _ = data_loader.load_test_data()

# 加载模型
model = Deeplab_v3_plus(final_filters=151, num_middle=8, img_size=256, input_channel=3,
                        aspp_filters=128, final_activation='softmax')
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.load_weights(checkpoint_save_path)

predict_list = model.predict(test_img_list, batch_size=12)
print(predict_list.shape)
print(predict_list[0][100][100], predict_list[1][200][200], predict_list[2][100][100])
print(predict_list[20][10][100], predict_list[40][250][200], predict_list[45][8][100])
print(predict_list[30][10][100], predict_list[50][200][3], predict_list[86][45][100])

predict_list = onehot_to_class(predict_list, mask_size=256)

print(predict_list[0][100][100], predict_list[1][200][200], predict_list[2][100][100])
print(predict_list[20][10][100], predict_list[40][250][200], predict_list[45][8][100])
print(predict_list[30][10][100], predict_list[50][200][3], predict_list[86][45][100])

# 获取预测图的文件名
test_label_list = glob.glob(test_label_file_path + '*.png')
img_name_list = []
for img_file in test_label_list:
    img_name = (img_file.split('\\')[-1])
    img_name_list.append(img_name)

num_test_list = []
for _ in range(13):
    num_test_list.append(0)

predict_img_list = []
for predict in predict_list:
    predict_img = np.ones(shape=(256, 256, 3), dtype=np.uint8)
    predict_img *= 128

    for row in range(256):
        for col in range(256):
            predict_img[row][col][2] = predict[row][col] * 10
            num_test_list[predict[row][col]] += 1

    predict_img_list.append(predict_img)

for predict_img, img_name in zip(predict_img_list, img_name_list):
    cv2.imwrite(predict_save_path + img_name, predict_img)

print(num_test_list)