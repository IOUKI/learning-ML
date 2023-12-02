import pandas as pd
import numpy as np
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

# xy: 真實數據, yPred: 預測數值
def computeCost(x, y, w, b):
    yPred = w * x + b
    # 成本函數: (真實數據 - 預測值)**2
    cost = (y - yPred) ** 2
    
    # 計算成本函數平均數值
    cost = cost.sum() / len(x)
    
    return cost

# 計算斜率
def computeGradient(x, y, w, b):
    wGradient = (x * (w * x + b - y)).mean() # mean: 計算平均
    bGradient = (w * x + b - y).mean()
    return wGradient, bGradient

# learningRate = 0.001 # 學習綠率

# 梯度下降
# x,y: 真實數據
# learningRate: 學習率
# runIter: 計算幾次
# pIter: 每幾圈就列印一次
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
            print(f'Ieration: {i:5}, Cost: {cost:.2e}, w: {w:.2e}, b: {b:.2e}, w gradient: {wGradient:.2e}, b gradient: {bGradient:.2e}')

    return w, b, wHist, bHist, cHist

wInit = 50
bInit = 50
learningRate = 1.0e-3
runIter = 100000
wFinal, bFinal, wHist, bHist, cHist = gradientDescent(x, y, wInit, bInit, learningRate, computeCost, computeGradient, runIter)

print(f'Final w: {wFinal:.2f}, b: {bFinal:.2f}')

print(f'年資3.5 預測薪資: {wFinal * 3.5 + bFinal:.1f}K')
print(f'年資5.9 預測薪資: {wFinal * 5.9 + bFinal:.1f}K')

# w = -100~100 b = -100~100 的cost
# arrange: 創建-100到101的矩陣
ws = np.arange(-100, 101)
bs = np.arange(-100, 101)

# zeros: 創建數值為0的矩陣，這邊創建了二為矩陣
costs = np.zeros((201, 201))

# 雙重迴圈尋遍所有w與b的組合結果
i = 0
for w in ws:
    j = 0
    for b in bs:
        cost = computeCost(x, y, w, b)
        costs[i, j] = cost
        j += 1
    i += 1

# 繪製3D圖表示所有結果
ax = plt.axes(projection='3d')
ax.xaxis.set_pane_color((0, 0, 0))
ax.yaxis.set_pane_color((0, 0, 0))
ax.zaxis.set_pane_color((0, 0, 0))

# 繪製3d網格
bGrid, wGrid = np.meshgrid(bs, ws)
ax.plot_surface(bGrid, wGrid, costs, cmap='Spectral_r', alpha=0.7)  # cmap: 網格顏色, alpha: 透明度
ax.plot_wireframe(wGrid, wGrid, costs, color='black', alpha=0.1)    # 網格線條

ax.set_title('w b 對應的 cost')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('cost')

# 取得最小cost位置
wIndex, bIndex = np.where(costs == np.min(costs))
ax.scatter(ws[wIndex], bs[bIndex], costs[wIndex, bIndex], color='red', s=40)
ax.scatter(wHist[0], bHist[0], cHist[0], color='green', s=40)
ax.plot(wHist, bHist, cHist)
print(f'當w={ws[wIndex]}和b={bs[bIndex]}會有最小cost: {costs[wIndex, bIndex]}')

plt.show()