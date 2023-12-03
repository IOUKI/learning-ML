from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 簡單線性模型
class LinearRegressionModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.rand(1, requires_grad=True))
        self.b = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, x):
        return self.w * x + self.b
    
def getData():
    data = pd.read_csv('./Salary_Data.csv')
    x = data['YearsExperience']
    y = data['Salary']

    # 將資料分成測試集和訓練集
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=87)
    
    # 轉換numpy array格式
    xTrain = xTrain.to_numpy()
    xTest = xTest.to_numpy()
    yTrain = yTrain.to_numpy()
    yTest = yTest.to_numpy()

    # 從numpy array格式轉換成torch tensor格式
    xTrain = torch.from_numpy(xTrain)
    xTest = torch.from_numpy(xTest)
    yTrain = torch.from_numpy(yTrain)
    yTest = torch.from_numpy(yTest)

    return xTrain, xTest, yTrain, yTest

def main():
    # 將資料分成測試集和訓練集
    xTrain, xTest, yTrain, yTest = getData()

    # 固定排序
    torch.manual_seed(87)

    # 使用模型預測資料
    model = LinearRegressionModel()
    yPred = model(xTest)
    # print(yPred, yTest)
    # 可以看到預測資料跟實際資料差的很多，因為w,b都是隨機的

    # 使用MSELoss計算cost數值
    costFunction = nn.MSELoss()
    # cost = costFunction(yPred, yTest)
    # print(model.state_dict())
    # print(cost)

    # 梯度下降作法
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # optimizer.zero_grad() # 斜率歸零
    # cost.backward() # 計算梯度
    # optimizer.step() # 更新梯度

    # yPred = model(xTest)
    # cost = costFunction(yPred, yTest)
    # print(model.state_dict())
    # print(cost)

    trainCostHist = []
    testCostHist = []

    running = 2000

    for epoch in range(running):
        
        # 訓練階段
        model.train() 

        yPred = model(xTrain) # 計算所有預測值

        trainCost = costFunction(yPred, yTrain) # 用預測數據與真實數據計算cost
        trainCostHist.append(trainCost.detach().numpy()) # 取消 tensor 追蹤梯度，tensor 轉換 numpy

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

        # Epoch: 看完一次所有訓練集稱為 1 個Epoch
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:5} Train cost: {trainCost: .4e} Test cost: {testCost: .4e}")

    # 顯示cost計算狀態
    def showCostHist():
        plt.plot(range(running), trainCostHist, label='train cost')
        plt.plot(range(running), testCostHist, label='test cost')
        plt.title('train & test cost')
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.legend()
        plt.show()
    # showCostHist()

    # 查看final w, b
    # print(model.state_dict())
    # print(model.w, model.b)

    # 查看預測數據與實際數據的差異
    model.eval()
    with torch.inference_mode():
        yPred = model(xTest)
    print(yPred, yTest)

    # 儲存訓練好的模型
    model.state_dict()
    torch.save(obj=model.state_dict(), f='./pytorchLinearRegression.pth')

if __name__ == '__main__':
    main()