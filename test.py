# import pandas as pd
# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import csv2png_train, csv2png_test
# from pre_process import load_fer2013

# faces, emotions = load_fer2013()
#
# print(faces)

# data = pd.read_csv('data/train.csv')
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_gen = train_datagen.flow_from_dataframe(dataframe=data, directory='data/', x_col='train.csv', y_col='pixels')
# print(train_gen)

# csv2png_train('./data/train.csv', 'train')
csv2png_test('./data/test.csv', 'test')

