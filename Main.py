import random
import random as rn
from numpy import matlib
import pandas as pd
from CMPA import CMPA
from EOO import EOO
from GSO import GSO
from Global_Vars import Global_Vars
from Model_BiLSTM import Model_BiLSTM
from Model_DTCNN import Model_DTCNN
from Model_HDAN import Model_HDAN
from Model_LSTM import Model_LSTM
from Model_MobileNet import Model_MobileNet
from Model_RBM import Model_RBM
from Plot_Results import *
from Proposed import Proposed
from WSA import WSA
from objfun import objfun

no_dataset = 2

#  Read the dataset 1
an = 0
if an == 1:
    Dataset = './Dataset/APA-DDoS-Dataset.csv'
    df = pd.read_csv(Dataset)
    df.drop('frame.time', inplace=True, axis=1)
    file = np.asarray(df)
    data = file[:, 2:-1]

    tar = file[:, -1]
    Uni = np.unique(tar)
    uni = np.asarray(Uni)
    Target = np.zeros((tar.shape[0], len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(tar == uni[j])
        Target[ind, j] = 1
    np.save('Data_1.npy', data)  # Save the Dataset_1
    np.save('Target_1.npy', Target)  # Save the Target_1

# Read the dataset 2
an = 0
if an == 1:
    Dataset = './Dataset/wustl-scada-2018.csv'
    df = pd.read_csv(Dataset)
    data_arr = np.asarray(df)
    zeroind = np.where(data_arr[:, 6] == 0)
    data_0 = data_arr[zeroind[0][:100000], :-1]
    tar_0 = data_arr[zeroind[0][:100000], -1]
    oneind = np.where(data_arr[:, 6] == 1)
    data_1 = data_arr[oneind[0][:100000], :-1]
    tar_1 = data_arr[oneind[0][:100000], -1]
    data = np.append(data_0, data_1, axis=0)
    tar = np.append(tar_0, tar_1, axis=0)

    index = np.arange(data.shape[0])
    random.shuffle(index)
    data = data[index, :]
    tar = tar[index]

    np.save('Data_2.npy', data)  # Save the Dataset_2
    np.save('Target_2.npy', np.reshape(tar, (-1, 1))) # Save the Target_2

# Data Cleaning
a = 0
if a == 1:
    for n in range(2):
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        df = pd.DataFrame(Data)
        Pre_data = df.dropna()  # Remove 0 Values
        Pre_data.drop_duplicates(inplace=True)  # Remove duplicates Values
        Pre_data.drop(Pre_data, inplace=True)  # Removing the outliers
        data_arr = np.asarray(Pre_data)
        np.save('Preprocess_Data_' + str(n + 1) + '.npy', data_arr)

# Weight feature Selection using RBM
an = 0
if an == 1:
    for n in range(no_dataset):
        data = np.load('Preprocess_Data_' + str(n + 1) + '.npy', allow_pickle=True)
        value = Model_RBM(data)
        np.save('Rbm_Feature_' + str(n + 1) + '.npy', value)

# optimal feature selection
an = 0
if an == 1:
    for n in range(no_dataset):
        Feat = np.load('Rbm_Feature_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Dataset
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 200
        xmin = matlib.repmat(np.append((1 * np.ones((1, Chlen - 100))), (0.01 * np.ones((1, Chlen - 100)))), Npop, 1)
        xmax = matlib.repmat(np.append((Feat.shape[1] * np.ones((1, Chlen - 100))), (0.09 * np.ones((1, Chlen - 100)))),
                             Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
        fname = objfun
        max_iter = 50

        print('EOO....')
        [bestfit1, fitness1, bestsol1, Time1] = EOO(initsol, fname, xmin, xmax, max_iter)  # EOO

        print('WSA....')
        [bestfit2, fitness2, bestsol2, Time2] = WSA(initsol, fname, xmin, xmax, max_iter)  # WSA

        print('GSO....')
        [bestfit3, fitness3, bestsol3, Time3] = GSO(initsol, fname, xmin, xmax, max_iter)  # GSO

        print('CMPA....')
        [bestfit4, fitness4, bestsol4, Time4] = CMPA(initsol, fname, xmin, xmax, max_iter)  # CMPA

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)  # Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BEST_Sol_' + str(n + 1) + '.npy', BestSol)

# Feature Selection
an = 0
if an == 1:
    for n in range(no_dataset):
        Feat = np.load('Rbm_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        bests = np.load('BEST_Sol_' + str(n + 1) + '.npy', allow_pickle=True).astype(int)
        sol = np.round(bests[4, :]).astype(np.int16)
        feat = []
        for j in range(Feat.shape[0]):
            F1 = Feat[j].astype('f')
            feat_1 = sol[100:] * F1[:, sol[:100]]
            feat.append(feat_1)
        np.save('Selected_Feature_' + str(n + 1) + '.npy', feat)

# Classification
an = 0
if an == 1:
    for m in range(no_dataset):
        Feature = np.load('Selected_Feature_' + str(m + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(m + 1) + '.npy', allow_pickle=True)  # loading step
        K = 5
        Per = 1 / 5
        Perc = round(Feature.shape[0] * Per)
        eval = []
        for i in range(K):
            Eval = np.zeros((5, 14))
            Feat = Feature
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval[0, :], pred = Model_MobileNet(Train_Data, Train_Target, Test_Data,
                                               Test_Target)  # Model MobileNet
            Eval[1, :], pred1 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)  # Model LSTM
            Eval[2, :], pred2 = Model_DTCNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model DTCN
            Eval[3, :], pred3 = Model_BiLSTM(Train_Data, Train_Target, Test_Data, Test_Target)  # Model Bi-LSTM
            Eval[4, :], pred4 = Model_HDAN(Train_Data, Train_Target, Test_Data, Test_Target)  # DTCN + LSTM
            eval.append(Eval)
    np.save('Eval_all.npy', eval)  # Save Eval

plotConvResults()
plot_results()
Plot_ROC_Curve()
