import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Attention
from Evaluation import evaluation


def Model_BiLSTM(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [0, 0, 4, 50]
    out, model3 = LSTM_Bi_train(train_data, train_target, test_data, test_target, sol)
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred


def LSTM_Bi_train(train_data, train_target, test_data, test_target, sol):
    # if sol is None:
    #     sol = [0, 0, 4, 50]
    Optimizers = ['adam', 'SGD', 'RMSProp', 'AdaDelta', 'Adagrad']
    Act = ['linear', 'sigmoid', 'relu', 'tanh']
    n_unique_words = 1000  # cut texts after this number of words
    # maxlen = 20
    # batch_size = 128
    # (train_data, train_target),(test_data, test_target) = imdb.load_data(num_words=n_unique_words)
    # x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    model = Sequential()
    model1 = Sequential()
    model2 = Sequential()
    model3 = Sequential()

    model.add(Embedding(n_unique_words, 128, input_length=(train_data.shape[1])))
    # model.add(LSTM(int(sol[0]), input_shape=(1, trainX.shape[2])))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    # Attention layer
    attention = Attention()
    model.attention()
    # Multi-scale Dense Layer
    model.add(Dense(1, activation=(Act[int(sol[1])])))
    model1.add(Dense(1, activation=(Act[int(sol[1])])))
    model2.add(Dense(1, activation=(Act[int(sol[1])])))
    model3 = (model + model1 + model2) / 3

    model3.compile(loss='binary_crossentropy', optimizer=(Optimizers[int(sol[1])]), metrics=['accuracy'])
    model3.fit(train_data, train_target,
               batch_size=int(sol[2]),
               epochs=int(sol[3]),
               validation_data=[test_data, test_target])
    testPredict = model3.predict(test_data)

    return testPredict, model3
