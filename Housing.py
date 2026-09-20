import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

Data = pd.read_csv('F:/AI-ML/mumbai.csv')

def train_test_split(Data_X, Data_Y):
    N = len(Data_X)

    X_Train = Data_X.loc[:int(N * 0.8)]
    X_Test = Data_X.loc[int(N * 0.8):]
    Y_Train = Data_Y.loc[:int(N * 0.8)]
    Y_Test = Data_Y.loc[int(N * 0.8):]

    return [X_Train, X_Test, Y_Train, Y_Test]

def sep_cat(Data_Y):
    col = Data_Y.columns
    String_Col = Data_Y[col].select_dtypes(include=['object', 'string']).columns.tolist()
    for i in String_Col:
        Data_Y[i] = Data_Y.groupby(i).ngroup()
    return Data_Y



Data_X = Data['price']
Data_Y = sep_cat(Data.drop('price', axis = 1)).drop(['title', 'locality', 'price_per_sqft', 'city'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(Data_X, Data_Y)
N = len(x_train)

X = np.hstack((np.ones((N, 1)), y_train.values))
print(X)

A = (np.linalg.inv((X.T) @ X) @ (X.T)) @ np.array(x_train)
print(A)

y = X @ A
print(y)

Intercept = A[0]
Slope = A[1:]

for i in y_train.columns:
    r = np.corrcoef(y_train[i], x_train)[0, 1]
    print(i, " : ", round(r, 3))

MSE = np.mean((np.array(x_train) - y) ** 2)
print("MSE: ", MSE)

plt.scatter(x_train, y)
plt.plot([x_train.min(), x_train.max()], [x_train.min(), x_train.max()], color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# inp = []
# for col in y_train.columns:
#     inp.append(float(input("Enter " + col + ": ")))
# x_new = np.hstack(([[1]], np.array([inp])))
# print("Predicted price:", x_new @ A)