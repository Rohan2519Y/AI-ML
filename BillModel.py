import numpy as np
import math

Total_Bill = np.array([34, 108, 64, 88, 99, 51])
Tip_Amount = np.array([5, 17, 11, 8, 14, 5])

TBsum = np.sum(Total_Bill)
TAsum = np.sum(Tip_Amount)
print('Sum of Total_Bill :', TBsum)
print('Sum of Tip_Amount :', TAsum)
TBsqr = np.sum(Total_Bill * Total_Bill)
TAsqr = np.sum(Tip_Amount * Tip_Amount)
print('Square of Total_Bill :', TBsqr)
print('Square of Tip_Amount :', TAsqr)
TATB = np.sum(Total_Bill * Tip_Amount)
print("TATB :", TATB)
R = (len(Total_Bill) * TATB - (TBsum * TAsum))/ (math.sqrt((len(Total_Bill) * TBsqr - (TBsum * TBsum)) * (len(Total_Bill) * TAsqr - (TAsum * TAsum))))
print("Corelation : ", R)

TBmean = np.mean(Total_Bill)
TAmean = np.mean(Tip_Amount)

b1 = np.sum((Total_Bill - TBmean) * (Tip_Amount - TAmean)) / np.sum((Total_Bill - TBmean) * (Total_Bill - TBmean))
b0 = TAmean - b1 * TBmean

print("Slope : ", b1)
print("Intercept : ", b0)

n = int(input("Enter Number : "))
y = b0 + b1 * n
print(y)