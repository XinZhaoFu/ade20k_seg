import shutil
from utils import recreate_dir
import numpy as np
import glob

ori_img_file_path = './data/ori-img/'
ori_label_file_path = './data/ori-label/'
dis_img_train_file_path = './data/train/img/'
dis_label_train_file_path = './data/train/label/'
dis_img_val_file_path = './data/val/img/'
dis_label_val_file_path = './data/val/label/'
dis_img_test_file_path = './data/test/img/'
dis_label_test_file_path = './data/test/label/'


def shuffle_file(img_file_list, label_file_list):
    """
    打乱img和label的文件列表顺序 并返回两列表
    :param img_file_list:
    :param label_file_list:
    :return:
    """
    np.random.seed(10)
    index = [i for i in range(len(img_file_list))]
    np.random.shuffle(index)
    img_file_list = np.array(img_file_list)[index]
    label_file_list = np.array(label_file_list)[index]
    return img_file_list, label_file_list


def distribution_file(dis_img_file_list, dis_label_file_list, dis_img_file_path, dis_label_file_path):
    for img_file, label_file in zip(dis_img_file_list, dis_label_file_list):
        img_name = img_file.split('\\')[-1]
        label_name = label_file.split('\\')[-1]
        shutil.copyfile(img_file, dis_img_file_path + img_name)
        shutil.copyfile(label_file, dis_label_file_path + label_name)
        print(img_file, label_file, dis_img_file_path + img_name, dis_label_file_path + label_name)


ori_img_file_list = glob.glob(ori_img_file_path + '*.jpg')
ori_label_file_list = glob.glob(ori_label_file_path + '*.png')
assert (len(ori_img_file_list) == len(ori_label_file_list))
shuffle_img_file_list, shuffle_label_file_list = shuffle_file(ori_img_file_list, ori_label_file_list)
# for ori_img_file, ori_label_file in zip(shuffle_img_file_list, shuffle_label_file_list):
#     print(ori_img_file, ori_label_file)

recreate_dir(dis_img_train_file_path)
recreate_dir(dis_label_train_file_path)
recreate_dir(dis_img_val_file_path)
recreate_dir(dis_label_val_file_path)
recreate_dir(dis_img_test_file_path)
recreate_dir(dis_label_test_file_path)

distribution_file(shuffle_img_file_list[:int(len(ori_img_file_list) * 0.6)],
                  shuffle_label_file_list[:int(len(ori_img_file_list) * 0.6)],
                  dis_img_train_file_path, dis_label_train_file_path)

distribution_file(shuffle_img_file_list[int(len(ori_img_file_list) * 0.6):int(len(ori_img_file_list) * 0.8)],
                  shuffle_label_file_list[int(len(ori_img_file_list) * 0.6):int(len(ori_img_file_list) * 0.8)],
                  dis_img_val_file_path, dis_label_val_file_path)

distribution_file(shuffle_img_file_list[int(len(ori_img_file_list) * 0.8):],
                  shuffle_label_file_list[int(len(ori_img_file_list) * 0.8):],
                  dis_img_test_file_path, dis_label_test_file_path)