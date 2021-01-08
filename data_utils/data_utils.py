from glob import glob
import numpy as np
import cv2
from utils import shuffle_file, write_hdf5
from gc import collect
import tensorflow as tf
from random import randint, shuffle, choice


tf_decode_jpeg = tf.image.decode_jpeg
tf_resize = tf.image.resize
tf_decode_png = tf.image.decode_png
tf_read_file = tf.io.read_file
tf_cast = tf.cast
tf_uint8 = tf.uint8
cv2_resize = cv2.resize


def get_img_mask_list(file_number,
                      file_path,
                      mask_size=512,
                      batch_size=16):
    """
    将图像和标签数据队列处理后以tensor返回
    图像格式为(size, size, 3)
    标签格式为(size, size, 1)
    标签总计151类(含背景)
    :param batch_size:
    :param file_number:可以节取一部分数据
    :param file_path:
    :param mask_size:
    :return:
    """
    autotune = tf.data.experimental.AUTOTUNE

    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_file_path_list = glob(img_path + '*.jpg')
    label_file_path_list = glob(label_path + '*.png')

    assert len(img_file_path_list) == len(label_file_path_list)

    # 截取部分文件
    if file_number > len(img_file_path_list):
        file_number = len(img_file_path_list)
    img_file_path_list, label_file_path_list = img_file_path_list[:file_number], label_file_path_list[:file_number]

    # 文件对应检查
    check_img_label_list(img_file_path_list, label_file_path_list)

    img_file_path_ds = tf.data.Dataset.from_tensor_slices(img_file_path_list)
    image_ds = img_file_path_ds.map(load_and_preprocess_image, num_parallel_calls=autotune)
    label_file_path_ds = tf.data.Dataset.from_tensor_slices(label_file_path_list)
    label_ds = label_file_path_ds.map(load_and_preprocess_label, num_parallel_calls=autotune)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # image_label_ds = image_label_ds.cache(filename='data/cache')
    image_label_ds = image_label_ds.shuffle(buffer_size=batch_size * 8)
    # image_label_ds = image_label_ds.repeat()
    image_label_ds = image_label_ds.batch(batch_size)
    image_label_ds = image_label_ds.prefetch(buffer_size=autotune)

    return image_label_ds


def get_img_mask_hdf5(file_path,
                      mask_size=256,
                      augmentation_mode=False,
                      augmentation_rate=1,
                      erase_rate=0.1):
    """
    将图像和标签存为hdf5文件
    真正的猛男敢于在服务器上使用这种方式一次性把所有文件都读取到内存里
    节省io时间 同时锻炼抗击打能力
    图像格式为(size, size, 3)
    标签格式为(size, size, 1)
    标签总计151类(含背景)
    :param erase_rate:
    :param augmentation_rate:
    :param file_path:
    :param mask_size:
    :param augmentation_mode:
    :return:
    """
    img_path = file_path + 'img/'
    label_path = file_path + 'label/'
    img_file_list = glob(img_path + '*.jpg')
    label_file_list = glob(label_path + '*.png')
    assert len(img_file_list) == len(label_file_list)

    # 图片补足 以batchsize为64为假想
    num_supplement = 64 * ((len(img_file_list) // 64) + 1) - len(img_file_list)
    img_file_supplement_list = img_file_list[:num_supplement]
    label_file_supplement_list = label_file_list[:num_supplement]
    img_file_list.extend(img_file_supplement_list)
    label_file_list.extend(label_file_supplement_list)

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    if augmentation_mode is True:
        img_array_hdf5 = np.empty(shape=(augmentation_rate * len(img_file_list), mask_size, mask_size, 3),
                                  dtype=np.float16)
        mask_array_hdf5 = np.empty(shape=(augmentation_rate * len(label_file_list), mask_size, mask_size, 1),
                                   dtype=np.uint8)
    else:
        img_array_hdf5 = np.empty(shape=(len(img_file_list), mask_size, mask_size, 3), dtype=np.float16)
        mask_array_hdf5 = np.empty(shape=(len(label_file_list), mask_size, mask_size, 1), dtype=np.uint8)

    file_index = 0
    cv2imread = cv2.imread
    cv2resize = cv2.resize
    npfloat16 = np.float16
    for img_file, label_file in zip(img_file_list, label_file_list):
        img = cv2imread(img_file)
        label = cv2imread(label_file)

        img_name = (img_file.split('\\')[-1]).split('.')[0]
        label_name = (label_file.split('\\')[-1]).split('.')[0]
        assert img_name == label_name

        if augmentation_mode is True:
            for img, label in hdf5_augmentation(img,
                                                label,
                                                mask_size=mask_size,
                                                erase_rate=erase_rate,
                                                augmentation_rate=augmentation_rate):
                img_array_hdf5[file_index, :, :, :] = img[:, :, :] / npfloat16(255.)
                mask_array_hdf5[file_index, :, :, 0] = label[:, :, 0]
                file_index += 1
        else:
            img = cv2resize(img, (mask_size, mask_size))
            label = cv2resize(label, (mask_size, mask_size))
            img_array_hdf5[file_index, :, :, :] = img[:, :, :] / np.float16(255.)
            mask_array_hdf5[file_index, :, :, 0] = label[:, :, 0]
            file_index += 1

        if file_index % 1000 == 0:
            print('已加载数目：\t' + str(file_index))

    write_hdf5(img_array_hdf5, file_path + 'img.hdf5')
    write_hdf5(mask_array_hdf5, file_path + 'mask.hdf5')

    del img_array_hdf5
    del mask_array_hdf5
    collect()


def load_and_preprocess_image(path):
    image = tf_read_file(path)
    image = tf_decode_jpeg(image, channels=3)
    image = tf_resize(image, [512, 512])
    image /= 255.0
    return image


def load_and_preprocess_label(path):
    image = tf_read_file(path)
    image = tf_decode_png(image)
    image = tf_resize(image, [64, 64])
    image = tf_cast(image, dtype=tf_uint8)

    return image


def check_img_label_list(img_list, label_list):
    print('文件对应检查')
    for img_path, label_path in zip(img_list, label_list):
        img_name = (img_path.split('\\')[-1]).split('.')[0]
        label_name = (label_path.split('\\')[-1]).split('.')[0]
        assert img_name == label_name
    print('文件对应检查通过')


def hdf5_augmentation(img,
                      label,
                      mask_size=256,
                      erase_rate=0.5,
                      augmentation_rate=1):
    """
    对图片随机引入了翻转旋转裁剪遮盖
    :param augmentation_rate:
    :param erase_rate:
    :param img:
    :param label:
    :param mask_size:
    :return:
    """
    assert mask_size > 0

    for i in range(augmentation_rate):

        # 如果数据扩增数量大于1 则除第一张以外的图片均进行裁剪 以保证图片互异
        if i:
            img, label = img_crop(img, label, mask_size)

        # 引入resize 如果没有经裁剪 使得图片尺寸变为规定尺寸 则该步骤进行统一修改
        width, length, channel = img.shape
        if width != mask_size or length != mask_size or img.shape != label.shape:
            img = cv2_resize(img, dsize=(mask_size, mask_size))
            label = cv2_resize(label, dsize=(mask_size, mask_size))

        # 水平方向的翻转
        flip_choice = choice([0, 1])
        if flip_choice:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)

        # 在0 90 180 270 之间任选一值进行随机旋转
        rotate_choice = choice([0, 1])
        if rotate_choice:
            img, label = img_rotate(img, label, rot_num=1, img_size=mask_size)

        # 在cutout或gridmask之间任选一种遮盖方法
        erase_choice = choice([0, 1])
        if erase_choice:
            cutout_gridmask_choice = choice([0, 1])
            if cutout_gridmask_choice:
                img = gridMask(img, rate=erase_rate, img_size=mask_size)
            else:
                img = cutout(img, rate=erase_rate, img_size=mask_size)

        yield img, label


def img_crop(ori_img,
             ori_label,
             crop_size):
    """
    随机裁剪该图的部分区域 并resize为制定大小
    :param ori_img:
    :param ori_label:
    :param crop_size:
    :return:
    """
    assert crop_size > 0

    resize_img = cv2_resize(ori_img, dsize=(2 * crop_size, 2 * crop_size))
    resize_label = cv2_resize(ori_label, dsize=(2 * crop_size, 2 * crop_size))

    random_point_x = randint(1, int(2 * crop_size - crop_size - 1))
    random_point_y = randint(1, int(2 * crop_size - crop_size - 1))

    crop_img = np.empty(shape=(crop_size, crop_size, 3), dtype=np.uint8)
    crop_label = np.empty(shape=(crop_size, crop_size, 3), dtype=np.uint8)

    crop_img[:, :, :] = resize_img[random_point_x:random_point_x + crop_size,
                        random_point_y:random_point_y + crop_size, :]
    crop_label[:, :, :] = resize_label[random_point_x:random_point_x + crop_size,
                          random_point_y:random_point_y + crop_size, :]

    return crop_img, crop_label


def img_rotate(ori_img,
               ori_label,
               rot_num=4,
               img_size=256):
    """
    图像及标注旋转 默认含四个方向的图像 0度 90度 180度 270度
    :param img_size:
    :param ori_img:待旋转图像
    :param ori_label:待旋转标注图像
    :param rot_num:希望多少张旋转图
    :return:所得图像列表，所得标注列表
    """
    rot_img_list = []
    rot_label_list = []
    rotated_list = [0, 90, 180, 270]

    if rot_num > 4 or rot_num <= 0:
        rot_num = 4
    if 0 < rot_num < 4:
        shuffle(rotated_list)
        rotated_list = rotated_list[:rot_num]

    for rotated in rotated_list:
        rotated_matrix = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), rotated, 1)

        rot_img_temp = cv2.warpAffine(ori_img, rotated_matrix, (img_size, img_size))
        rot_img_temp = np.reshape(rot_img_temp, ori_img.shape)

        rot_label_temp = cv2.warpAffine(ori_label, rotated_matrix, (img_size, img_size))
        rot_label_temp = np.reshape(rot_label_temp, ori_label.shape)

        if rot_num == 1:
            return rot_img_temp, rot_label_temp

        rot_img_list.append(rot_img_temp)
        rot_label_list.append(rot_label_temp)

    return rot_img_list, rot_label_list


def cutout(ori_img,
           rate=0.5,
           img_size=256):
    """
    对正方形图片进行cutout 遮盖位置随机
        长方形需要改一下
    遮盖比例为空时用默认值图像尺寸的一半作为遮盖的边长 即默认遮盖四分之一区域
        使用经验，过拟合遮盖率增大，欠拟合去掉遮盖，具体数值自行调节
    添加遮盖前 对图像一圈进行0填充
    :param img_size:
    :param ori_img: 输入应为正方形图像
    :param rate: cutout的遮盖图形设定为正方形，该变量为其边长与图像边长的比例
    :return:cutout后的图像
    """
    mask_length = int(img_size * rate)
    region_x, region_y = randint(0, int(img_size + mask_length)), randint(0, int(img_size + mask_length))

    fill_img = np.zeros((int(img_size + mask_length * 2), int(img_size + mask_length * 2), 3))
    fill_img[int(mask_length):int(mask_length + img_size), int(mask_length):int(mask_length + img_size)] = ori_img
    fill_img[region_x:int(region_x + mask_length), region_y:int(region_y + mask_length)] = 0
    cutout_img = fill_img[int(mask_length):int(mask_length + img_size), int(mask_length):int(mask_length + img_size)]

    return cutout_img


def gridMask(ori_img,
             rate=0.5,
             img_size=256):
    """
    对图片进行gridmask 每行每列各十个 以边均匀十等分 每一长度中包含mask长度、offset偏差和留白
        长方形需要改一下
    其余可参考cutout的注释
    :param img_size:
    :param ori_img: 输入应为正方形图像
    :param rate: mask长度与十分之一边长的比值
    :return: gridmask后的图像
    """
    fill_img_length = int(img_size + 0.2 * img_size)
    offset = randint(0, int(0.1 * fill_img_length))
    mask_length = int(0.1 * fill_img_length * rate)
    fill_img = np.zeros((fill_img_length, fill_img_length, 3))
    fill_img[int(0.1 * img_size):int(0.1 * img_size) + img_size,
    int(0.1 * img_size):int(0.1 * img_size) + img_size] = ori_img
    for width_num in range(10):
        for length_num in range(10):
            length_base_patch = int(0.1 * fill_img_length * length_num) + offset
            width_base_patch = int(0.1 * fill_img_length * width_num) + offset
            fill_img[length_base_patch:length_base_patch + mask_length,
            width_base_patch:width_base_patch + mask_length, ] = 0
    gridmask_img = fill_img[int(0.1 * img_size):int(0.1 * img_size) + img_size,
                   int(0.1 * img_size):int(0.1 * img_size) + img_size]

    return gridmask_img


def resize_img_label_list(img_list,
                          label_list,
                          mask_size):
    """
    对传入的img列表和label列表统一尺寸
    :param img_list:
    :param label_list:
    :param mask_size:
    :return:
    """
    resize_img_list, resize_label_list = [], []
    for img, label in zip(img_list, label_list):
        img = cv2_resize(img, dsize=(mask_size, mask_size))
        label = cv2_resize(label, dsize=(mask_size, mask_size))
        resize_img_list.append(img)
        resize_label_list.append(label)

    return resize_img_list, resize_label_list
