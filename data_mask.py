import glob
import scipy.io as scio
import numpy as np
import cv2


def get_colors_dict():
    label_path = './data/ori-label/'
    label_list = glob.glob(label_path + '*.png')
    print(len(label_list))
    colors_dict = {}
    num_dict = 1
    num_file = 1

    for label_file in label_list:
        if num_file>10:
            break
        print(num_file, label_file)
        num_file = num_file + 1
        label = cv2.imread(label_file)
        width, length, _ = label.shape
        for w in range(width):
            for l in range(length):
                point_color = label[w][l]
                tuple_point_color = (point_color[0], point_color[1])
                if tuple_point_color not in colors_dict:
                    dict = {tuple_point_color: num_dict}
                    num_dict = num_dict + 1
                    colors_dict.update(dict)
    print('-----------')
    print(num_dict-1)
    return colors_dict


def get_color_mask(label, colors_dict):
    width, length, _ = label.shape
    mask = np.empty(shape=(width, length))
    for w in range(width):
        for l in range(length):
            point_color = label[w][l]
            tuple_point_color = (point_color[0], point_color[1], point_color[2])
            mask[w][l] = colors_dict.get(tuple_point_color, 0)
    return mask


colors_dict = get_colors_dict()
print(colors_dict)

