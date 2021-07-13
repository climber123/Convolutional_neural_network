import os
import cv2
import numpy as np
import keras
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model
from google.colab.patches import cv2_imshow


def preprocess(img):
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img


def loadData(basePath):
    labelsTrain = []
    ptrnsTrain = []

    path = basePath + "/class0"
    for fileName in os.listdir(path):
        img = cv2.imread(path + '/' + fileName)
        img = preprocess(img)
        ptrnsTrain.append(img)
        labelsTrain.append(0)

    path = basePath + "/class1"
    for fileName in os.listdir(path):
        img = cv2.imread(path + '/' + fileName)
        img = preprocess(img)
        ptrnsTrain.append(img)
        labelsTrain.append(1)

    ptrnsTrain = np.array(ptrnsTrain)
    labelsTrain = np.array(labelsTrain)

    rndState = np.random.get_state()
    np.random.set_state(rndState)
    np.random.shuffle(ptrnsTrain)
    np.random.set_state(rndState)
    np.random.shuffle(labelsTrain)

    return (ptrnsTrain, labelsTrain)


def createModel(wsize):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, wsize, wsize)
    else:
        input_shape = (wsize, wsize, 3)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(8, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Conv2D(8, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    # opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['accuracy'])  # optimizer=opt, loss='categorical_crossentropy'
    model.summary()
    return model


def trainClassifier(modelName, pathTrain, pathTest):
    (ptrnsTrain, labelsTrain) = loadData(pathTrain)
    ptrnsTrain = ptrnsTrain.reshape(ptrnsTrain.shape[0], wsize, wsize, 3)
    labelsTrain = keras.utils.to_categorical(labelsTrain, numlasses)
    history = model.fit(ptrnsTrain, labelsTrain, batch_size=40, epochs=25)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.save(modelName)

    (ptrnsTest, labelsTest) = loadData(pathTest)
    ptrnsTest = ptrnsTest.reshape(ptrnsTest.shape[0], wsize, wsize, 3)
    labelsTest = keras.utils.to_categorical(labelsTest, numlasses)
    evaluateRes = model.evaluate(ptrnsTest, labelsTest, verbose=0)
    print('Test loss:\t', evaluateRes[0])
    print('Test accuracy:\t', evaluateRes[1])


def testClassifier(modelName, imgName):
    model = load_model(modelName)
    img = cv2.imread(imgName)
    width = img.shape[1]
    height = img.shape[0]
    blocksW = int(width / wsize)
    blocksH = int(height / wsize)

    for x in range(blocksW):
        for y in range(blocksH):
            imgRoi = img[y * wsize:y * wsize + wsize, x * wsize:x * wsize + wsize]
            imgRoi = preprocess(imgRoi)
            ptrnTestX = []
            ptrnTestX.append(imgRoi)
            ptrnTestX = np.array(ptrnTestX)
            ptrnTestX = ptrnTestX.reshape(ptrnTestX.shape[0], wsize, wsize, 3)

            prediction = model.predict(ptrnTestX)
            # print('prediction ', prediction)
            if prediction[0, 0] > prediction[0, 1]:
                cv2.rectangle(img, (x * wsize + 1, y * wsize + 1), ((x + 1) * wsize - 1, (y + 1) * wsize - 1),
                              color=(0, 255, 255), thickness=2)
    print('Marked')
    cv2_imshow(img)


numlasses = 2
wsize = 50
pathTrain = "sample_data/train"
pathTest = "sample_data/test"
modelName = "sample_data/classifier.h5"

model = createModel(wsize)

trainClassifier(modelName, pathTrain, pathTest)

testClassifier(modelName, 'sample_data/00076418.jpg')
