'''
将progress.csv中的数据保留两位小数，并输出到progress_trunc.csv
'''
import os
import pandas as pd

# 遍历所有以data_开头的文件夹，包括子文件夹
for root, dirs, files in os.walk('.'):
    for folder in dirs:
            folder_path = os.path.join(root, folder)
            csv_path = os.path.join(folder_path, 'progress.csv')
            if os.path.isfile(csv_path):
                # 读取CSV
                df = pd.read_csv(csv_path)
                # 保留两位小数
                df = df.round(2)
                # 输出到progress_trunc.csv
                out_path = os.path.join(folder_path, 'progress_trunc.csv')
                df.to_csv(out_path, index=False)