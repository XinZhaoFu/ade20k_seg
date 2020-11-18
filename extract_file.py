import glob

import cv2
import os
import shutil

origin_file_path = 'E:\\datasets\\ADE20K_2016_07_26\\images'
target_img_file_path = './data/ori-img/'
target_label_file_path = './data/ori-label/'


def findFile(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                img_name = (file_path.split('\\')[-1]).split('.')[0]
                file_path_split = file_path.split('ADE_')[0]
                img_path = file_path_split + img_name + '.jpg'
                mask_path = file_path_split + img_name + '_seg.png'
            yield img_path, mask_path, img_name


def main():
    target_img_file_list = glob.glob(target_img_file_path + '*.jpg')
    target_label_file_list = glob.glob(target_label_file_path + '*.png')
    print(len(target_img_file_list), len(target_label_file_list))
    if len(target_img_file_list) == 0 or len(target_label_file_list) == 0 or len(target_img_file_list) != len(
            target_label_file_list):
        for ori_img_path, ori_mask_path, img_name in findFile(origin_file_path):
            target_img_path = target_img_file_path + img_name + '.jpg'
            target_label_path = target_label_file_path + img_name + '_seg.png'
            shutil.copyfile(ori_img_path, target_img_path)
            shutil.copyfile(ori_mask_path, target_label_path)
            print(img_name)
    else:
        print('无需重新抽取')


if __name__ == '__main__':
    main()
