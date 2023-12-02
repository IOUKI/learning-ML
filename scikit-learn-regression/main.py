"""
在scikit中有許多做好的模型提供給大家使用
以下是2,3章使用sklearn模型製作的方法
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 邏輯回歸
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

# 使用sklearn內建的邏輯回歸類別
lgModel = LogisticRegression()
lgModel.fit(xTrain, yTrain)
yPred = lgModel.predict(xTest)
acc = (yPred == yTest).sum() / len(yPred) * 100 # 計算準確率
print(f"糖尿病 正確率: {acc}%")

# ----------------------------------------------------------------------
# 多元線性回歸
data = pd.read_csv('./Salary_Data2.csv')

# label encoding: 如果特徵存在高低關係，可以使用label encoding做分類
data['EducationLevel'] = data['EducationLevel'].map({'高中以下': 0, '大學': 1, '碩士以上': 2})

# one-hot encoding: 如果特徵不存在高低關係，可以使用one-hot encoding做分類，可以省略最後一個特徵來節省分類
# 使用sklearn的OneHotEncoder來自動分類沒有高低關係的特徵
onehotEncoder = OneHotEncoder()
onehotEncoder.fit(data[['City']])
cityEncoded = onehotEncoder.transform(data[['City']]).toarray()
data[['CityA', 'CityB', 'CityC']] = cityEncoded

# 移除多餘的資料和特徵
# axis: 1是列, 0是行
data = data.drop(['City', 'CityC'], axis=1)
# print(data)

# 訓練集、測試集
# 將資料分成兩份: 一份拿來測試、一份拿來訓練
# 通常會分測試20%、訓練80%
x = data[['YearsExperience', 'EducationLevel', 'CityA', 'CityB']]
y = data['Salary']

# test_size: 指定測試集為20%訓練集為80%, random_state: 固定資料排序
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=87)

# 將pandas的資料轉換成numpy的格式以便後續計算
xTrain = xTrain.to_numpy()
xTest = xTest.to_numpy()

lrModel = LinearRegression()
lrModel.fit(xTrain, yTrain)
yPred = lrModel.predict(xTest)

print('薪資預測:')
print(pd.DataFrame({
    'yPred': yPred,
    'yTest': yTest
}))