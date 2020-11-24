import os
import shutil
import h5py
import numpy as np


def create_dir(folder_name):
    """
    创建文件夹
    :param folder_name:
    :return:
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def recreate_dir(folder_name):
    """
    重建文件夹
    :param folder_name:
    :return:
    """
    shutil.rmtree(folder_name)
    create_dir(folder_name)


def load_hdf5(in_file_path):
    """
    载入hdf5文件
    :param in_file_path:
    :return:返回该文件
    """
    with h5py.File(in_file_path, "r") as f:
        return f["image"][()]


def write_hdf5(data, out_file_path):
    """
    写入hdf5文件
    :param data:
    :param out_file_path:
    :return:
    """
    with h5py.File(out_file_path, "w") as f:
        f.create_dataset("image", data=data, dtype=data.dtype)


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
