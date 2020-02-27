import cv2
import numpy as np
import pandas as pd
import os
import logging

IMG_SIZE_X = 48
IMG_SIZE_Y = 48
DIR_PREFIX = './data/'
logging.basicConfig(level=logging.DEBUG)


# 将csv中的信息保存为图片
#  0 anger 生气
#  1 disgust 厌恶
#  2 fear 恐惧
#  3 happy 开心
#  4 sad 伤心
#  5 surprised 惊讶
#  6 normal 中性
def csv2png_train(csv_path, img_dir_name):
    # 检测是否存在图片文件夹
    try:
        os.mkdir(DIR_PREFIX + img_dir_name)
        for each in range(7):
            os.mkdir(DIR_PREFIX + img_dir_name + '/' + str(each))
    # 为了确保文件夹是空的
    except FileExistsError:
        raise Exception(f'已存在目标文件夹，请删除目标文件夹 {DIR_PREFIX + img_dir_name + "/"} 再执行转换操作')

    count_list = [0, 0, 0, 0, 0, 0, 0]
    data = pd.read_csv(csv_path)
    emotion_list = data['emotion']
    img_list = data['pixels']
    # print(len(img_list[0]))
    # cv2.imwrite('tmp.png', img_list[0].split(' ').reshape(IMG_SIZE_X, IMG_SIZE_Y))
    for each in range(len(emotion_list)):
        img_path_and_filename = str(emotion_list[each])+'/'+str(count_list[int(emotion_list[each])])+'.png'
        face = [int(pixel) for pixel in img_list[each].split(' ')]
        face = np.asarray(face).reshape(IMG_SIZE_X, IMG_SIZE_Y)
        cv2.imwrite(DIR_PREFIX + img_dir_name + '/' + img_path_and_filename, face)
        logging.info(f'已转换图片文件{img_path_and_filename} 保存于{DIR_PREFIX+img_dir_name+"/"}')
        count_list[int(emotion_list[each])] += 1


def csv2png_test(csv_path, img_dir_name):
    # 检测是否存在图片文件夹
    try:
        os.mkdir(DIR_PREFIX+img_dir_name)
    # 为了确保文件夹是空的
    except FileExistsError:
        raise Exception(f'已存在目标文件夹，请删除目标文件夹 {DIR_PREFIX + img_dir_name + "/"} 再执行转换操作')

    data = pd.read_csv(csv_path)
    img_list = data['pixels']
    # print(len(img_list[0]))
    # cv2.imwrite('tmp.png', img_list[0].split(' ').reshape(IMG_SIZE_X, IMG_SIZE_Y))
    for each in range(len(img_list)):
        img_filename = str(each) + '.png'
        face = [int(pixel) for pixel in img_list[each].split(' ')]
        face = np.asarray(face).reshape(IMG_SIZE_X, IMG_SIZE_Y)
        cv2.imwrite(DIR_PREFIX + img_dir_name + '/' + img_filename, face)
        logging.info(f'已转换图片文件{img_filename} 保存于{DIR_PREFIX+img_dir_name+"/"}')


def csv2png_icml(csv_path):
    # 检测是否存在图片文件夹
    try:
        os.mkdir(DIR_PREFIX + 'icml_' + 'train')
        for each in range(7):
            os.mkdir(DIR_PREFIX + 'icml_' + 'train/' + str(each))

        os.mkdir(DIR_PREFIX + 'icml_' + 'public_test')
        for each in range(7):
            os.mkdir(DIR_PREFIX + 'icml_' + 'public_test/' + str(each))

        os.mkdir(DIR_PREFIX + 'icml_' + 'private_test')
        for each in range(7):
            os.mkdir(DIR_PREFIX + 'icml_' + 'private_test/' + str(each))
    # 为了确保文件夹是空的
    except FileExistsError:
        raise Exception(f'请确认 data 目录下不存在 icml_train icml_public_test icml_private_test 文件夹后再执行转换操作')

    data_set = pd.read_csv(csv_path)
    usage_list = data_set[' Usage'].tolist()
    emotion_list = data_set['emotion'].tolist()
    pixels_list = data_set[' pixels']

    count_dict = {'Training': [0, 0, 0, 0, 0, 0, 0],
                  'PublicTest': [0, 0, 0, 0, 0, 0, 0],
                  'PrivateTest': [0, 0, 0, 0, 0, 0, 0]}
    dir_name_dict = {'Training': 'icml_train',
                     'PublicTest': 'icml_public_test',
                     'PrivateTest': 'icml_private_test'}
    for each in range(data_set.shape[0]):
        tmp_usage = usage_list[each]
        tmp_emotion = emotion_list[each]
        tmp_pixels = np.asarray([int(pixel) for pixel in pixels_list[each].split(' ')]).reshape(IMG_SIZE_X, IMG_SIZE_Y)

        cv2.imwrite(DIR_PREFIX
                    + dir_name_dict[tmp_usage] + '/'
                    + str(tmp_emotion) + '/'
                    + str(count_dict[tmp_usage][tmp_emotion]) + '.png'
                    , tmp_pixels)

        logging.info(f"已转换图片文件{str(count_dict[tmp_usage][tmp_emotion]) + '.png'} "
                     f"保存于{dir_name_dict[tmp_usage] + '/' + str(tmp_emotion) + '/'}")

        count_dict[tmp_usage][tmp_emotion] += 1

