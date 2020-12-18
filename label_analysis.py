import cv2
from glob import glob
import numpy as np


def analysis(label_path_list):
    analysis_list = np.zeros(shape=151)
    for i in range(len(label_path_list)):
        label_path = label_path_list[i]
        print(i, label_path)
        label = cv2.imread(label_path)
        row, col, channel = label.shape
        label_array = np.empty(shape=(row, col), dtype=np.int64)
        label_array[:, :] = label[:, :, 0]

        for r in range(row):
            for c in range(col):
                point_class = label_array[r, c]
                analysis_list[point_class] += 1

    return analysis_list


label_file_path = 'data/ori_annotation/'
label_file_list = glob(label_file_path + '*.png')
np.random.shuffle(label_file_list)
label_file_list = label_file_list[:200]
print(len(label_file_list))

analysis_result = analysis(label_file_list)
np.savetxt('./data/class_analysis.csv', analysis_result, fmt='%d', delimiter=',')
