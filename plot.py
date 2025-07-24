import matplotlib.pyplot as plt
import numpy as np
import os
from feasytools import ReadOnlyTable, DTypeEnum

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['font.size'] = 20  # 设置全局字号
plt.rcParams['axes.titlesize'] = 24  # 设置标题字号
plt.rcParams['axes.labelsize'] = 24  # 设置坐标轴标签字号
plt.rcParams['legend.fontsize'] = 20  # 设置图例字号
plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度字号
plt.rcParams['ytick.labelsize'] = 20  # 设置y轴刻度字号

def plot(xmean, xmax, xmin, xtestmean, xtestmax, xtestmin, title, xlabel, ylabel, baseline, save_to):
    x = np.arange(1, len(xmean)+1, 1)
    plt.figure(figsize=(10, 6),tight_layout=True)
    plt.fill_between(x, xmin, xmax, color='blue', alpha=0.2, label='训练回合回报范围')
    plt.plot(x, xmean, color='blue', linewidth=2, label='训练回合回报均值')
    #plt.fill_between(x, xtestmin, xtestmax, color='red', alpha=0.2, label='Test Min-Max Range')
    plt.plot(x, xtestmean, color='red', linewidth=2, label='测试回合回报')
    if baseline is not None:
        plt.axhline(y=baseline, color='green', linestyle='--', linewidth=2, label='Baseline')
    plt.legend()
    plt.title(title)
    plt.xlim(1, len(xmean))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_to)

def find_progress_csv(directory):
    progress_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "progress.csv":
                dir = os.path.join(root, file)
                tb = ReadOnlyTable(dir, dtype=DTypeEnum.FLOAT32)
                bsres = root + "/baseline_results.csv"
                print(bsres)
                bs = None
                if os.path.exists(bsres):
                    tb2 = ReadOnlyTable(bsres)
                    bs = np.mean(tb2.col("epret"))
                plot(
                    tb.col("AverageEpRet"), tb.col("MaxEpRet"), tb.col("MinEpRet"),
                    tb.col("AverageTestEpRet"), tb.col("MaxTestEpRet"), tb.col("MinTestEpRet"),
                    "",
                    "轮次", "回合回报", bs,
                    dir.replace(".csv", ".png")
                )
                progress_files.append(os.path.join(root, file))
    return progress_files

directory_to_search = "."
result = find_progress_csv(directory_to_search)