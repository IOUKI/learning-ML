# 分類問題
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./Diabetes_Data.csv')

# 資料轉換
# 男生: 1, 女生: 0
data['Gender'] = data['Gender'].map({'男生': 1, '女生': 0})

x = data[['Age', 'Weight', 'BloodSugar', 'Gender']]
y = data['Diabetes']

# 分類訓練集和測試集
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=87)
xTrain = xTrain.to_numpy()
xTest = xTest.to_numpy()

# 特徵縮放
scaler = StandardScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

# S型函數
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# w = np.array([1, 2, 3, 4])
# b = 1
# z = (w * xTrain).sum(axis=1) + b
# print(sigmoid(z))

# Cost Function - Binary Cross Entropy 二元交叉熵
# cost = -y * log(yPred) - (1 - y) * log(1 - yPred)
def computeCost(x, y, w, b):
    z = (w * x).sum(axis=1) + b
    yPred = sigmoid(z)
    cost = -y * np.log(yPred) - (1 - y) * np.log(1 - yPred)
    cost = cost.mean()
    return cost

# w = np.array([1, 4, 2, 4])
# b = 2
# print(computeCost(xTrain, yTrain, w, b))

# optimizer - gradient descent = 根據斜率改變參數
# z = (w * xTrain).sum(axis=1) + b
# yPred = sigmoid(z)
# bGradient = (yPred - yTrain).mean()
# wGradient = np.zeros(xTrain.shape[1])

# for i in range(xTrain.shape[1]):
#     wGradient[i] = (xTrain[:, i] * (yPred - yTrain)).mean()

# print(wGradient, bGradient)

def computeGradient(x, y, w, b):
    z = (w * x).sum(axis=1) + b
    yPred = sigmoid(z)
    wGradient = np.zeros(xTrain.shape[1])
    bGradient = (yPred - y).mean()

    for i in range(xTrain.shape[1]):
        wGradient[i] = (xTrain[:, i] * (yPred - y)).mean()

    return wGradient, bGradient

# 梯度下降
np.set_printoptions(formatter={'float': '{: .2e}'.format})
def gradientDescent(x, y, wInit, bInit, learningRate, costFunction, gradientFunction, runIter, pIter=1000):

    # 紀錄cost, w, b
    cHist = []
    wHist = []
    bHist = []

    # 初始 w, b
    w = wInit
    b = bInit
    for i in range(runIter):
        wGradient, bGradient = gradientFunction(x, y, w, b)
        w = w - wGradient * learningRate
        b = b - bGradient * learningRate
        cost = costFunction(x, y, w, b)

        cHist.append(cost)
        wHist.append(w)
        bHist.append(b)

        # 每一千次print資料
        if i % pIter == 0:
            print(f'Ieration: {i:5}, Cost: {cost:.2e}, w: {w}, b: {b:.2e}, w gradient: {wGradient}, b gradient: {bGradient:.2e}')

    return w, b, wHist, bHist, cHist

wInit = np.array([1, 2, 2, 3])
bInit = 5
learningRate = 1
runIter = 10000
wFinal, bFinal, wHist, bHist, cHist = gradientDescent(xTrain, yTrain, wInit, bInit, learningRate, computeCost, computeGradient, runIter)

# 繪出cost數值變化
# plt.plot(np.arange(0, 100), cHist[:100])
# plt.title('iteration vs cost')
# plt.xlabel('iteration')
# plt.ylabel('cost')
# plt.show()

# 繪出w數值變化
# plt.plot(np.arange(0, 1000), wHist[:1000])
# plt.title('iteration vs w')
# plt.xlabel('iteration')
# plt.ylabel('w')
# plt.show()

# 繪出b數值變化
# plt.plot(np.arange(0, 1000), bHist[:1000])
# plt.title('iteration vs b')
# plt.xlabel('iteration')
# plt.ylabel('b')
# plt.show()

# 測試
z = (wFinal * xTest).sum(axis=1) + bFinal
yPred = sigmoid(z)
yPred = np.where(yPred > 0.5, 1, 0) # 當y pred 中的直大於0.5 將結果轉成1 否則為0
acc = (yPred == yTest).sum() / len(yPred) * 100 # 計算準確率
print(f"正確率: {acc}%")

# 套用到真實狀況
# 72 92 102 女生
# 62 52 120 男生
xReal = np.array([[72, 92, 102, 0], [62, 52, 120, 1]])
xReal = scaler.transform(xReal)
z = (wFinal * xReal).sum(axis=1) + bFinal
yReal = sigmoid(z)
print(yReal)