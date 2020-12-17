import cv2
from glob import glob
import numpy as np


def analysis(label_path_list):
    analysis_list = np.zeros(shape=12 * 255 + 255)
    for i in range(len(label_path_list)):
        label_path = label_path_list[i]
        print(i, label_path)
        label = cv2.imread(label_path)
        row, col, channel = label.shape
        label_array = np.empty(shape=label.shape, dtype=np.int)
        label_array[:, :, :] = label[:, :, :]

        for r in range(row):
            for c in range(col):
                point_class = int(label_array[r, c, 2] / 10 * 255 + label_array[r, c, 1])
                analysis_list[point_class] += 1

    num_non_zero = 0
    for i in np.nditer(analysis_list):
        if i:
            num_non_zero += 1

    analysis_non_zero = np.empty(shape=(num_non_zero, 2), dtype=np.int)
    i_analysis_non_zero = 0
    for i in range(len(analysis_list)):
        if analysis_list[i]:
            analysis_non_zero[i_analysis_non_zero, 0] = i
            analysis_non_zero[i_analysis_non_zero, 1] = analysis_list[i]
            i_analysis_non_zero += 1

    return analysis_non_zero


label_file_path = './data/ori-label/'
label_file_list = glob(label_file_path + '*.png')
np.random.shuffle(label_file_list)
label_file_list = label_file_list[:100]
print(len(label_file_list))

analysis_result = analysis(label_file_list)
np.savetxt('./data/analysis.csv', analysis_result, fmt='%d', delimiter=',')
