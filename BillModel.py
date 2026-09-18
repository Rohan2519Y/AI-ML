import numpy as np
import math
from MyLinearRegression import LinearRegression
# from sklearn.linear_model import LinearRegression

Total_Bill = np.array([34, 108, 64, 88, 99, 51])
Tip_Amount = np.array([5, 17, 11, 8, 14, 5])

Model = LinearRegression(Total_Bill, Tip_Amount)
Model.CC()
Model.slope()
Model.intersept()
Model.Predict(Total_Bill)
print(Model.new_pred)
M = Model.MSE()
print(M)
Model.drawRegression()



#########################################


# amount = Total_Bill.reshape(len(Total_Bill), 1)
# tip = Tip_Amount.reshape(len(Tip_Amount), 1)
# model = LinearRegression()
# model.fit(amount, tip)
# print(model.intercept_)
# print(model.coef_)
# print(model.predict([[200], [300]]))