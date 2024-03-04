import pandas as pd
from tqdm import tqdm

data = pd.read_csv('./preprocess_caption.csv')
fil_path = 'output.txt'
length = len(data)
k=0
part_length = 2
part =length //part_length

for j in range(part_length):
    if j==part_length - 1:
        with open (str(j) +fil_path, 'w') as file:
            for i in tqdm(data.caption[k:]):
                i=i.strip()
                if i[-1] =='\"' or i[-1] =='\'':
                    file.writelines(i[:-1] +'\n\n')
                else:
                    file.writelines(i +'\n\n')
        continue

    with open (str(j) +fil_path, 'w') as file:
        for i in tqdm(data.caption[k:k+part]):
            i=i.strip()
            if i[-1] =='\"' or i[-1] =='\'':
                file.writelines(i[:-1] +'\n\n')
            else:
                file.writelines(i +'\n\n')
    k = k+part
