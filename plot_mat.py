import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['font.size'] = 14

a = np.zeros((5,5), dtype=np.float32)
with open("data_drl2cs/search_results.csv") as fp:
    fp.readline()
    for ln in fp:
        pr, ret = ln.strip().split(",")
        p1, p2 = map(int, map(float, pr.split(";")))
        a[p1-1][p2-1] = float(ret)
        
plt.imshow(a, cmap='cool', interpolation='nearest')
plt.colorbar()
plt.xlabel('充电站2价格 (元/kWh)')
plt.ylabel('充电站1价格 (元/kWh)')
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        plt.text(j, i, f"{a[i, j]:.2f}", ha='center', va='center', color='black')
plt.xticks(np.arange(a.shape[1]), list(map(str,np.arange(1, a.shape[1]+1, dtype=np.float32))))
plt.yticks(np.arange(a.shape[0]), list(map(str,np.arange(1, a.shape[1]+1, dtype=np.float32))))
plt.show()