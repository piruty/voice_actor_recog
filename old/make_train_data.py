from pathlib import Path
import numpy as np
import csv

import make_mfcc_data


file_pathes = Path('./voice/remove_silence/').glob('**/*.wav')
data = []
labels = []

for file_path in file_pathes:
    mfcc = make_mfcc_data.convert_to_mfcc(str(file_path))
    for m in mfcc:
        data.append(np.append(m, file_path.stem))

with open('./files/data.txt', 'w') as data_f:
    writer = csv.writer(data_f)
    for d in data:
        writer.writerow(d)
