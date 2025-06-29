from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from feasytools import ReadOnlyTable, DTypeEnum

def plot(xmean, xmax, xmin, xtestmean, xtestmax, xtestmin, title, xlabel, ylabel, baseline, save_to):
    x = np.arange(1, len(xmean)+1, 1)
    plt.figure(figsize=(10, 6))
    plt.fill_between(x, xmin, xmax, color='blue', alpha=0.2, label='Train Min-Max Range')
    plt.plot(x, xmean, color='blue', linewidth=2, label='Train Mean')
    plt.fill_between(x, xtestmin, xtestmax, color='red', alpha=0.2, label='Test Min-Max Range')
    plt.plot(x, xtestmean, color='red', linewidth=2, label='Test Mean')
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
                    "Parameter with Mean, Max, and Min",
                    "Epoch", "Return", bs,
                    dir.replace(".csv", ".png")
                )
                progress_files.append(os.path.join(root, file))
    return progress_files

directory_to_search = "./data"
result = find_progress_csv(directory_to_search)