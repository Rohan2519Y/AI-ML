import pandas as pd

df = pd.read_csv('F:/AI-ML/mumbai.csv')

T = df['furnished'].unique()
# print(T)

mapping = {}
for i, v in enumerate(T):
    mapping[v] = i

df['furnished'] = df['furnished'].map(mapping)
print(df['furnished'])