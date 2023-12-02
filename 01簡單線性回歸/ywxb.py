import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.font_manager import fontManager

# 引入中文字體
fontManager.addfont('ChineseFont.ttf')
mlp.rc('font', family='ChineseFont')

data = pd.read_csv('./Salary_Data.csv')
# print(data)

# 數學上可以用 y = w * x + b 來表示一條直線
# 月薪 = w * 年資 + b
x = data['YearsExperience']
y = data['Salary']

def plotPred(w, b):
    # yPred:「預測值」數據，根據w和b的數值做出變動
    yPred = x * w + b 

    # plot: 依照數據繪出直線
    plt.plot(x, yPred, color='blue', label='預測線')

    # scatter: 依照數據繪出點陣圖
    plt.scatter(x, y, marker='x', color='red', label='真實數據')

    plt.title('年資-薪水')
    plt.xlabel('年資')
    plt.ylabel('薪水(千)')
    plt.xlim([0, 12])       # x軸最大最小值
    plt.ylim([-60, 140])    # y軸最大最小值
    plt.legend()            # 顯示label
    plt.show()

plotPred(w=10, b=20)