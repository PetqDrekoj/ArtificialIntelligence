from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


class Repository:
    # load train and test dataset
    def load_dataset(self):
        # load dataset
        (trainX, trainY), (testX, testY) = mnist.load_data()
        trainX = trainX[:10000, :, :]
        testX = testX[:10000, :, :]
        trainY = trainY[:10000]
        testY = testY[:10000]
        # reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
        return trainX, trainY, testX, testY


class Controller:
    def __init__(self, repository):
        self.repository = repository

    # scale pixels
    def prep_pixels(self, train, test):
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm

    # define cnn model
    def define_model(self):
        model = Sequential()
        # convolutional layer
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        # flatten output of conv
        model.add(Flatten())
        # hidden layer
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        # output layer
        model.add(Dense(10, activation='softmax'))
        # compiling the sequential model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # evaluate a model using k-fold cross-validation
    def evaluate_model(self, dataX, dataY, n_folds=5):
        scores = list()
        # prepare cross validation
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        # enumerate splits
        for train_ix, test_ix in kfold.split(dataX):
            # define model
            model = self.define_model()
            # select rows for train and test
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
            # fit model
            model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=0)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            print('> %.3f' % (acc * 100.0))
            # stores scores
            scores.append(acc)
        return scores


class UserInterface:
    def __init__(self, controller):
        self.controller = controller

    # run the test harness for evaluating a model
    def run(self):
        # load dataset
        trainX, trainY, testX, testY = self.controller.repository.load_dataset()
        # prepare pixel data
        trainX, testX = self.controller.prep_pixels(trainX, testX)
        # evaluate model
        scores = self.controller.evaluate_model(trainX, trainY)
        # show performance
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))


def main():
    repository = Repository()
    controller = Controller(repository)
    userinterface = UserInterface(controller)
    userinterface.run()


main()
