import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from Evaluation import evaluation


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def LSTM_train(trainX, trainY, testX, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(int(sol[0]), input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    testPredict = np.zeros((testX.shape[0], trainY.shape[1])).astype('int')
    for i in range(trainY.shape[1]):
        print(i)
        model.fit(trainX, trainY[:, i].reshape(-1, 1), epochs=round(sol[1]), batch_size=1, verbose=2)
        testPredict[:, i] = model.predict(testX).ravel()
    return testPredict, model


def Model_LSTM(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [3, 1]
    out, model = LSTM_train(train_data, train_target, test_data, sol)
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(out, test_target)
    # return np.asarray(Eval)[:, 0]
    return Eval, pred
