import os
import cv2
import logging
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# basic setting
logging.basicConfig(level=logging.INFO)

# global setting
IMG_SIZE_X = 48
IMG_SIZE_Y = 48
ACTIVATION_CHOICE = ['softmax', 'elu', 'selu',' softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']


def test_the_model(model, test_dir):
    """
    test the performance and print the accuracy of the model
    :param model: the model which will be tested
    :param test_dir: the directory's location which contains the test image set
                     the test images should be classified with the same name of the classification
    :return:
    """
    # the variable to calculate the global accuracy
    total = 0
    correct = 0
    correct_rate_list = []  # the different expression's accuracy

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # read the different image group in with the classification
    for each_emotion_group in range(7):
        # the variable to calculate the respective accuracy
        tmp_total = 0
        tmp_correct = 0

        # read the each of the images from the image group
        for each_img_num in range(len(os.listdir(test_dir + '/' + str(each_emotion_group) + '/'))):
            test_img = cv2.imread(test_dir + '/' + str(each_emotion_group) + '/' + str(each_img_num) + '.png', cv2.IMREAD_GRAYSCALE)
            test_img = np.expand_dims(np.asarray(test_img), 0)
            test_img = np.expand_dims(test_img, 1)

            # predict the expresion with the chosen model
            prediction = model.predict(test_img)
            # return the most likely expression
            max_index = int(np.argmax(prediction))
            # if the result is correct, increase the counter
            if each_emotion_group == max_index:
                logging.info(f"测试组 {str(each_emotion_group)} 中 图片 {str(each_img_num) + '.png'} 识别正确")
                correct += 1
                tmp_correct += 1
            # else the result is incorrect, print the wrong result
            else:
                logging.info(f"测试组 {str(each_emotion_group)} 中 图片 {str(each_img_num) + '.png'} 识别错误, 识别结果为 {emotion_dict[max_index]}")
            total += 1
            tmp_total += 1
        correct_rate_list.append(tmp_correct / tmp_total * 100)

        for each in correct_rate_list:
            print(f'{emotion_dict[correct_rate_list.index(each)]} 的识别率为 {each} %')
        print(f'总体识别率为: {correct / total * 100} %')


def load_model(model_name='model.h5'):
    """
    load a trained model
    :param model_name: the model's name which will be loaded
    :return: a trained model
    """
    model = create_model()
    model.load_weights(model_name)
    return model


def train_model(model,
                num_train,
                num_val,
                batch_size=64,
                num_epoch=30,
                train_dir='data/train',
                val_dir='data/test',
                model_name='model.h5'):
    """
    train a model and save it
    :param model: the model to train
    :param num_train: the number of the training set
    :param num_val: the number of the validation set
    :param batch_size: how many images input at a time for training
    :param num_epoch: how many times to training repeatedly
    :param train_dir: where is the train set(directory of the images with classification)
    :param val_dir: where is the validation set(directory of the images with classification)
    :param model_name: the name of the file which has saved the model
    :return: a trained model
    """
    # data generator for training and validation(provide the data)
    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE_X, IMG_SIZE_Y),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )

    validation_generator = validation_data_gen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE_X, IMG_SIZE_Y),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )

    # train the model
    start_time = datetime.datetime.now()

    model.compile(loss='categorical_corssentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size
    )

    end_time = datetime.datetime.now()
    interval = (end_time - start_time).seconds
    minutes = interval // 60
    seconds = interval % 60
    print(f"训练时间: {minutes} 分 {seconds} 秒")

    # draw the changes of the Accuracy and the Loss in the training process
    plot_model_history(model_info)
    model.save_weights(model_name)

    return model


def create_model():
    """
    create a model with some parameter
    :return: the initialized model
    """
    # create a model with layered structure
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE_X, IMG_SIZE_Y, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model


def plot_model_history(model_history):
    """
    draw the curve of the Accuracy and the Loss with the model trained history
    :param model_history: the curve's data resource
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # draw the accuracy curve
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1),
                      len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # draw the loss curve
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()
