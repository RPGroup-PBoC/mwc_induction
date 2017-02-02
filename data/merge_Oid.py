import glob
import pandas as pd

files = glob.glob('*Oid*microscopy*foldchange.csv')

with open('datasets_ignore.csv', 'r') as f:
    ignored_files = f.readlines()

dfs = []
for i in range(len(files)):
    print(i)
    if files[i] not in ignored_files:
        data = pd.read_csv(files[i], comment='#')
        if 'r2' in files[i]:
            data.insert(0, 'run_number', 2)
        elif 'r3' in files[i]:
            data.insert(0, 'run_number', 3)
        else:
            data.insert(0,'run_number', 1)
        dfs.append(data)

merged = pd.concat(dfs)
merged.to_csv('merged_Oid_data_foldchange.csv', index=False)
