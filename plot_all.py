from pathlib import Path
from typing import List, Sequence
import matplotlib.pyplot as plt
import numpy as np
import os
from feasytools import ReadOnlyTable, DTypeEnum

def plot(xtests:List[Sequence[float]], legends:List[str], title, xlabel, ylabel, baseline, save_to):
    x = np.arange(1, len(xtests[0])+1, 1)
    plt.figure(figsize=(10, 6))
    for d, l in zip(xtests, legends):
        plt.plot(x, d, linewidth=2, label=l)
    if baseline is not None:
        plt.axhline(y=baseline, color='green', linestyle='--', linewidth=2, label='search best')
    plt.legend()
    plt.title(title)
    plt.xlim(1, len(xtests[0]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_to)

def find_progress_csv(directory):
    progress_files = []
    tests = []
    legends = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "progress.csv":
                dir = os.path.join(root, file)
                tb = ReadOnlyTable(dir, dtype=DTypeEnum.FLOAT32)
                tests.append(tb.col("AverageTestEpRet"))
                legends.append(os.path.basename(os.path.dirname(root)))
                progress_files.append(os.path.join(root, file))
    
    best = os.path.join(directory, "search_results.csv")
    if not os.path.exists(best):
        print(f"Warning: {best} does not exist. Skipping baseline comparison.")
        baseline = None
    else:
        with open(best, "r") as f:
            f.readline()  # Skip the header
            ans = -1e9
            for line in f:
                ans = max(ans, float(line.strip().split(",")[1]))
        baseline = ans
    
    plot(
        xtests=tests,
        legends=legends,
        title="Test Returns Comparison",
        xlabel="Epoch",
        ylabel="Return",
        baseline=baseline,
        save_to=os.path.join(directory, "test_returns_comparison.png")
    )
    return progress_files

directory_to_search = "./data"
result = find_progress_csv(directory_to_search)