from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import torch

# 邏輯回歸模型
class LogisticRegressionModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linearLayer = nn.Linear(in_features=4, out_features=1, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linearLayer(x))

data = pd.read_csv('./Diabetes_Data.csv')

# 資料轉換
# 男生: 1, 女生: 0
data['Gender'] = data['Gender'].map({'男生': 1, '女生': 0})

x = data[['Age', 'Weight', 'BloodSugar', 'Gender']]
y = data['Diabetes']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=87)
xTrain = xTrain.to_numpy()
xTest = xTest.to_numpy()
yTrain = yTrain.to_numpy()
yTest = yTest.to_numpy()

# 2023/12/03: 忘記特徵縮放
scaler = StandardScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

xTrain = torch.from_numpy(xTrain)
xTest = torch.from_numpy(xTest)
yTrain = torch.from_numpy(yTrain)
yTest = torch.from_numpy(yTest)

# 重製資料shape
yTrain = yTrain.reshape(-1, 1)
yTest = yTest.reshape(-1, 1)

# 設定結果的type
yTrain = yTrain.type(torch.double)
yTest = yTest.type(torch.double)

model = LogisticRegressionModel()
# print(model(xTrain))

costFunction = nn.BCELoss()

yPred = model(xTrain)
cost = costFunction(yPred, yTrain)
print(model.state_dict())
print(costFunction(yPred, yTrain))

# 梯度下降
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
optimizer.zero_grad()
cost.backward()
optimizer.step()

yPred = model(xTrain)
cost = costFunction(yPred, yTrain)
print(model.state_dict())
print(costFunction(yPred, yTrain))

trainCostHist = []
testCostHist = []
trainAccHist = []
testAccHist = []

running = 10000

for epoch in range(running):
    
    # 訓練階段
    model.train() 

    yPred = model(xTrain) # 計算所有預測值

    trainCost = costFunction(yPred, yTrain) # 用預測數據與真實數據計算cost
    trainCostHist.append(trainCost.detach().numpy()) # 取消 tensor 追蹤梯度，tensor 轉換 numpy
    
    trainAcc = (torch.round(yPred) == yTrain).sum() / len(yTrain) * 100
    trainAccHist.append(trainAcc.detach().numpy())

    optimizer.zero_grad() # 斜率歸零

    trainCost.backward() # 計算梯度

    optimizer.step() # 更新梯度

    # 測試階段
    model.eval()
    # 設定測試階段不要追蹤梯度狀態
    with torch.inference_mode():
        testPred = model(xTest)
        testCost = costFunction(testPred, yTest)
        testCostHist.append(testCost.detach().numpy())
        testAcc = (torch.round(testPred) == yTest).sum() / len(yTest) * 100
        testAccHist.append(testAcc.detach().numpy())

    # Epoch: 看完一次所有訓練集稱為 1 個Epoch
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:5} Train cost: {trainAcc}% Test cost: {testAcc}%")

# plt.plot(range(running), trainCostHist, label='train cost')
# plt.plot(range(running), testCostHist, label='test cost')
# plt.title('train & test cost')
# plt.xlabel('epochs')
# plt.ylabel('cost')
# plt.legend()
# plt.show()

# plt.plot(range(running), trainAccHist, label='train cost')
# plt.plot(range(running), testAccHist, label='test cost')
# plt.title('train & test cost')
# plt.xlabel('epochs')
# plt.ylabel('cost')
# plt.legend()
# plt.show()

print(model.state_dict())
model.eval()
with torch.inference_mode():
    yPred = model(xTest)
print((torch.round(yPred) == yTest).sum() / len(yTest) * 100)