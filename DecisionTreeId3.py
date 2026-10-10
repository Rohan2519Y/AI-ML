import pandas as pd
import numpy as np

Data = pd.DataFrame({
    "Day": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"],
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temp": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
})

def entropy(col, df=None):
    if df is None:
        df = Data
    frame = pd.DataFrame({col.name: col.values, 'last': df.iloc[:, -1].values}).reset_index(drop=True)
    N = len(frame)
    t = 0
    print(col.name)
    for v, g in frame.groupby(col.name):
        p = g['last'].value_counts(normalize=True)
        e = -sum(pi * np.log2(pi) for pi in p)
        t += len(g) / N * e
        print(f"{v}: entropy = {e:.3f}, counts = {dict(g['last'].value_counts())}")
    print(f"{col.name} conditional entropy: {t:.3f}\n")
    return t
    # df = df.groupby(col.name).size()
    # for i in df:
    #     print(df[i])
    # p = df / len(Data)
    # print(f"{col.name} Entropy : ", round(-sum(pi * np.log2(pi) for pi in p), 3))


E_Full_Data = Data.groupby(Data.iloc[:, -1]).size().reset_index(name='Count')
E_Full_Data["Probability"] = E_Full_Data["Count"] / len(Data)
E_Full_Data_Entropy = round((-(E_Full_Data['Count'][0] / len(Data)) * np.log2(E_Full_Data['Count'][0] / len(Data)) -(E_Full_Data['Count'][1] / len(Data)) * np.log2(E_Full_Data['Count'][1] / len(Data))), 2)

for i in Data.columns[1:-1]:
    entropy(Data[i])
# for i in Data.columns:
#     print(Data[i].unique)
# print(Data['Outlook'])
# print(Data)