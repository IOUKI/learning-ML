from sklearn.model_selection import train_test_split
from torch import nn
from linearRegression import getData
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 簡單線性模型
class LinearRegressionModel2(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # in_features: 特徵數量
        self.linearLayer = nn.Linear(in_features=1, out_features=1, dtype=torch.float64)

    def forward(self, x):
        return self.linearLayer(x)
    
xTrain, xTest, yTrain, yTest = getData()

model2 = LinearRegressionModel2()
# print(model2)
# print(model2.state_dict())

# 使用MSELoss計算cost數值
costFunction = nn.MSELoss()

# 重製資料shape
xTrain = xTrain.reshape(-1, 1)
yTrain = yTrain.reshape(-1, 1)
xTest = xTest.reshape(-1, 1)
yTest = yTest.reshape(-1, 1)

# yPred = model2(xTrain)
# cost = costFunction(yPred, yTrain)
# print(model2.state_dict())
# print(cost)

# # 梯度下降作法
optimizer = torch.optim.SGD(params=model2.parameters(), lr=0.01)
# optimizer.zero_grad() # 斜率歸零
# cost.backward() # 計算梯度
# optimizer.step() # 更新梯度

# yPred = model2(xTrain)
# cost = costFunction(yPred, yTrain)
# print(model2.state_dict())
# print(cost)

trainCostHist = []
testCostHist = []

running = 1000

for epoch in range(running):
    
    # --- 訓練階段 ---
    model2.train() 

    yPred = model2(xTrain) # 計算所有預測值

    trainCost = costFunction(yPred, yTrain) # 用預測數據與真實數據計算cost
    trainCostHist.append(trainCost.detach().numpy()) # 取消 tensor 追蹤梯度，tensor 轉換 numpy

    optimizer.zero_grad() # 斜率歸零

    trainCost.backward() # 計算梯度

    optimizer.step() # 更新梯度

    # --- 測試階段 ---
    model2.eval()
    # 設定測試階段不要追蹤梯度狀態
    with torch.inference_mode():
        testPred = model2(xTest)
        testCost = costFunction(testPred, yTest)
        testCostHist.append(testCost.detach().numpy())

    # Epoch: 看完一次所有訓練集稱為 1 個Epoch
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:5} Train cost: {trainCost: .4e} Test cost: {testCost: .4e}")

# 查看訓練結果
print(model2.state_dict())
print(torch.cuda.is_available())
print(torch.cuda.device_count())