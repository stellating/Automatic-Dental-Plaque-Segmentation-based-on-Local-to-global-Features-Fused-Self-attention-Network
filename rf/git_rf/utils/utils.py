import scipy.io as scio
import scipy.misc as misc
import numpy as np
import h5py
import os
from sklearn.ensemble import RandomForestClassifier
import math
import time
import cv2
import pickle
import matplotlib.pyplot as pyplot

def multi2onehot(y_train):
    """

    :param y_train:
    :return:
    """
    one_y = np.empty([0, 3])
    zero = np.array([1, 0, 0])
    one = np.array([0, 1, 0])
    two = np.array([0, 0, 1])
    for value in y_train:
        if value[0] == 0:
            one_y = np.vstack((one_y, zero))
        elif value[0] == 1:
            one_y = np.vstack((one_y, one))
        elif value[0] == 2:
            one_y = np.vstack((one_y, two))

    y_train = np.squeeze(y_train)
    # print('one y \n', one_y)
    # print('Success Load Data!', y_train.shape)
    # print('Success Load Data!', y_train)

    from sklearn.utils.multiclass import type_of_target

    print(type_of_target(y_train))
    print(type(y_train))
    print(type_of_target(one_y))
    print(type(one_y))

    return y_train, one_y



def normalize(x_list, max_list, min_list):
    """
    normalize the x_list
    :param x_list: the raw data of training sets
    :param max_list: the max list of raw data
    :param min_list: the min list of raw data
    :return: the normalized list
    """
    index = 0
    scalar_list = []
    for x in x_list:
        x_max = max_list[index]
        x_min = min_list[index]
        if x_max == x_min:
            x = 1.0
        else:
            x = np.round((x - x_min) / (x_max - x_min), 4)
        scalar_list.append(x)
        index += 1
    return scalar_list


def norm_data(x_train_nor):
    """
    normalize the train data and the test data
    :param x_train_nor: data of features for the normalization
    :param y_train_nor: data of tags for the normalization
    :return: the normalized features and normal labels
    """
    data_array = np.asmatrix(x_train_nor)
    max_list = np.max(data_array, axis=0)
    min_list = np.min(data_array, axis=0)

    scalar_data_mat = []
    for row_i in range(0, len(data_array)):
        if row_i == 0:
            row = data_array[row_i]
            row = row.tolist()
            scalar_data_mat.append(row)
        if row_i != 0:
            row = data_array[row_i]
            row = row.tolist()
            # print('row_i, row: ', row_i, row)
            scalar_row = normalize(row, max_list, min_list)
            scalar_data_mat.append(scalar_row)

    scalar_data_mat_np = np.array(scalar_data_mat)

    return scalar_data_mat_np


# load on mat file, return hks, deep_featrue
def load_x_file(filepath, is_norm=False):
    # data = h5py.File(filepath,'r')
    data = scio.loadmat(filepath)

    img_struct = data['img_struct'][0][0]

    hks = img_struct['hks'][0][0]
    deep_feature = img_struct['deep_feature'][0][0]

    x = img_struct['X'][0][0]

    y = img_struct['Y'][0][0]

    xy = np.hstack((x, y))

    r = img_struct['R'][0][0]

    g = img_struct['G'][0][0]

    b = img_struct['B'][0][0]

    rgb = np.hstack((r, g, b))

    if is_norm:
        print("normal ", xy[0:9])
        xy = norm_data(xy)
        print("normal ", xy[0:9])

    point = img_struct['Point'][0][0]  # 1* 1024

    # print("point ====",point[0][3][0].shape)
    len_p = len(point[0])
    # print("len ==",len_p)
    point_array = []

    for i in range(0, len_p):
        temp = point[0][i]
        point_array.append(temp)

    # x_train = np.hstack((xy, rgb, hks, deep_feature))
    x_train = np.hstack((xy, rgb, deep_feature))
    # x_train = deep_feature

    # generate_x_train(hks, deep_feature)
    return xy, rgb, hks, deep_feature, point_array, x_train


# def load_y_file(filepath):
#     data = scio.loadmat(filepath)
#     label_struct = data['label_struct'][0][0]
#     label = label_struct['label'][0][0]
#     return label

def load_y_file(filepath):

    data =  h5py.File(filepath,'r')
    img_struct = data['label_struct'][0][0]
    label = data[img_struct]['label']
    label = np.transpose(label)
    return label


def load_files(root_path_x, root_path_y, is_norm=False):
    xys = np.empty([0, 2])
    rgbs = np.empty([0, 3])
    hkss = np.empty([0, 23])
    deep_featrues = np.empty([0, 512])
    # deep_featrues = np.empty([0, 3])
    # x_trains = np.empty([0, 31]) #need to fix!
    # x_trains = np.empty([0, 540])  # need to fix!
    # x_trains = np.empty([0, 512])  # need to fix!
    x_trains = np.empty([0, 517])  # need to fix!
    # x_trains = np.empty([0, 3])  # need to fix!
    y_trains = np.empty([0, 1])
    for file in os.listdir(root_path_x):

        x_file_path = root_path_x + "/" + file
        y_file_path = root_path_y + "/" + file

        print('x_file_path:', x_file_path)
        print('y_file_path:', y_file_path)

        # get hks deep_featrue x_train
        xy, rgb, hks, deep_featrue, point, x_train = load_x_file(x_file_path, is_norm)
        y_train = load_y_file(y_file_path)
        # x_train = generate_x_train(hks, deep_featrue)

        # col cat
        x_trains = np.vstack((x_trains, x_train))

        xys = np.vstack((xys, xy))
        rgbs = np.vstack((rgbs, rgb))
        hkss = np.vstack((hkss, hks))
        deep_featrues = np.vstack((deep_featrues, deep_featrue))
        y_trains = np.vstack((y_trains, y_train))

    return x_trains, y_trains


def load_files_from_text(text_path, root_path_x, root_path_y, is_norm=False):
    xys = np.empty([0, 2])
    rgbs = np.empty([0, 3])
    hkss = np.empty([0, 23])
    deep_featrues = np.empty([0, 512])
    # deep_featrues = np.empty([0, 3])
    # x_trains = np.empty([0, 31]) #need to fix!
    # x_trains = np.empty([0, 540])  # need to fix!
    # x_trains = np.empty([0, 512])  # need to fix!
    x_trains = np.empty([0, 517])  # need to fix!
    # x_trains = np.empty([0, 3])  # need to fix!
    y_trains = np.empty([0, 1])
    text_file = open(text_path, 'r')
    for file_ in text_file.readlines():
    # for file in os.listdir(root_path_x):
    #     print('file_ is: ', file_)
        file = file_.strip('\n')
        x_file_path = root_path_x + "/" + file + '.mat'
        y_file_path = root_path_y + "/" + file + '.mat'

        print('x_file_path:', x_file_path)
        print('y_file_path:', y_file_path)

        # get hks deep_featrue x_train
        xy, rgb, hks, deep_featrue, point, x_train = load_x_file(x_file_path, is_norm)
        y_train = load_y_file(y_file_path)
        # x_train = generate_x_train(hks, deep_featrue)

        # col cat
        x_trains = np.vstack((x_trains, x_train))

        xys = np.vstack((xys, xy))
        rgbs = np.vstack((rgbs, rgb))
        hkss = np.vstack((hkss, hks))
        deep_featrues = np.vstack((deep_featrues, deep_featrue))
        y_trains = np.vstack((y_trains, y_train))

    return x_trains, y_trains


