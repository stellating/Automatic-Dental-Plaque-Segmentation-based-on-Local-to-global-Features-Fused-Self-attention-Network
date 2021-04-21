#! D:\Users\zhennanpang\Anaconda3\envs python
# -*- coding: utf-8 -*-
# @Time : 2020/8/20 16:47
# @Author : ZhennanpPang
# @File : aotu_param_main.py
# @Software: PyCharm

# from utils import utils
# from auto_param_gbdt_lr import compute_params
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
from utils import utils

def generate_x_train(hks, deep_feature):
    x_train = np.hstack((hks, deep_feature))
    return x_train



def train_test_get():
    # load tra
    data_train_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/JPEGImages_3_train'
    label_data_train_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/all_label_train_new_version'
    x_train_, y_train_ = utils.load_files(data_train_path, label_data_train_path, is_norm = False)
    y_train_= np.squeeze(y_train_)

    print('Success Load Data!', x_train_.shape)
    print('Success Load Data!', x_train_[0].shape)

    # load test
    data_test_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/JPEGImages_3_test'
    lable_data_test_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/all_label_train_new_version'
    x_test_, y_test_ = utils.load_files(data_test_path, lable_data_test_path, is_norm = False)
    y_test_ = np.squeeze(y_test_)

    print('Success Load Data!', x_train_.shape)
    print('Success Load Data!', x_train_[0].shape)

    return x_train_, y_train_, x_test_, y_test_


def get_train():
    # load tra
    data_train_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/JPEGImages_3_train'
    label_data_train_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/all_label_train_new_version'
    x_train_, y_train_ = utils.load_files(data_train_path, label_data_train_path, is_norm=False)
    y_train_ = np.squeeze(y_train_)

    print('Success Load Data!', x_train_.shape)
    print('Success Load Data!', x_train_[0].shape)
    return x_train_, y_train_


def get_test():
    # load test
    data_test_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/JPEGImages_3_test_small'
    lable_data_test_path = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/all_label_train_new_version'
    x_test_, y_test_ = utils.load_files(data_test_path, lable_data_test_path, is_norm=False)
    y_test_ = np.squeeze(y_test_)

    print('Success Load Data!', x_test_.shape)
    print('Success Load Data!', y_test_[0].shape)

    return x_test_, y_test_


def train_process():
    x_train_, y_train_, x_test_, y_test_ = train_test_get()
    # model = compute_params(x_train_, y_train_)
    # print('best model:', model)
    rf = RandomForestClassifier(n_estimators=100)
    # rf = RandomForestClassifier(n_estimators=200)
    rf.fit(x_train_,y_train_)
    print(rf.predict(x_train_))

    current_model_path = './model/'
    if not os.path.isdir(current_model_path):
        os.mkdir(current_model_path)
    current_model_name = 'rf_clf_100all_100_cnn3.pickle'

    with open(current_model_path+current_model_name, 'wb') as f:
        pickle.dump(rf, f)
        print('model saved successfully!')
        f.close()

from sklearn.metrics import accuracy_score


def test_process():
    # x_train_, y_train_, x_test_, y_test_ = train_test_get()
    x_test_, y_test_ = get_test()
    current_model_path = './model/'
    current_model_name = 'rf_clf_80_150.pickle'
    with open(current_model_path + current_model_name, 'rb') as f:
        rf = pickle.load(f)
        y_pred_ = rf.predict(x_test_)
        print(accuracy_score(y_test_, y_pred_))
        f.close()



if __name__ == '__main__':

    #load_x_file("/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3_small/cnn3_train/2018_000021.mat")
    # train_process()
    # print('train success! model saved!')
    test_process()

    # model = compute_params(x_train_, one_y)
    localtime = time.asctime(time.localtime(time.time()))
    print("Finished Time:", localtime)


