import os
import shutil


def create_dir(folder_name):
    """
    创建文件夹
    :param folder_name: 文件夹列表
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