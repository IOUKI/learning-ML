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
# print(computeCost(x, y, 10, 10))

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
# print(costs)

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
print(f'當w={ws[wIndex]}和b={bs[bIndex]}會有最小cost: {costs[wIndex, bIndex]}')

plt.show()