import pandas as pd

# Script that generates an train_full dataset where trains_small is held out

in_path = 'data/processed/'
out_path = 'data/hold_out/'

full = pd.read_csv(in_path + 'train_full.csv')
small = pd.read_csv(in_path + 'train_small.csv')
result = full[~full['tweet'].isin(small['tweet'])]
result.to_csv(out_path + 'train_full.csv', index=False, sep=',')
small.to_csv(out_path + 'train_small.csv', index=False, sep=',')