import matplotlib.pyplot as plt
import numpy as np
import os
from feasytools import ReadOnlyTable, DTypeEnum

def plot(xmean, xmax, xmin, xtestmean, xtestmax, xtestmin, title, xlabel, ylabel, save_to):
    x = np.arange(0, len(xmean), 1)
    plt.figure(figsize=(10, 6))
    plt.fill_between(x, xmin, xmax, color='blue', alpha=0.2, label='Train Min-Max Range')
    plt.plot(x, xmean, color='blue', linewidth=2, label='Train Mean')
    plt.fill_between(x, xtestmin, xtestmax, color='red', alpha=0.2, label='Test Min-Max Range')
    plt.plot(x, xtestmean, color='red', linewidth=2, label='Test Mean')
    plt.legend()
    plt.title(title)
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
                plot(
                    tb.col("AverageEpRet"), tb.col("MaxEpRet"), tb.col("MinEpRet"),
                    tb.col("AverageTestEpRet"), tb.col("MaxTestEpRet"), tb.col("MinTestEpRet"),
                    "Parameter with Mean, Max, and Min",
                    "Epoch", "Return",
                    dir.replace(".csv", ".png")
                )
                progress_files.append(os.path.join(root, file))
    return progress_files

# 示例：在当前目录及其子目录中查找 progress.csv 文件
directory_to_search = "."  # 当前目录
result = find_progress_csv(directory_to_search)