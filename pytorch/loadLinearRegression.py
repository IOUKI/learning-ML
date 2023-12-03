# 載入訓練好的模型
from linearRegression import LinearRegressionModel, getData
import torch

xTrain, xTest, yTrain, yTest = getData()

model = LinearRegressionModel()
model.state_dict()

# 載入訓練好的模型
model.load_state_dict(torch.load(f='./pytorchLinearRegression.pth'))

print(model.state_dict())

# 比對數據
model.eval()
with torch.inference_mode():
    yPred = model(xTest)
print(yPred, yTest)