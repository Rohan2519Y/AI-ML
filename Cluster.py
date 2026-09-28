import numpy as np
import math

M1 = 4
M2 = 11

Data = np.array([2, 4, 10, 12, 3, 20, 30, 11, 25])
Cluster = []
M1_Cluster = np.array([])
M2_Cluster = np.array([])
while True:
    P1, P2 = M1, M2
    C1 = []
    C2 = []
    M1_Cluster = abs(Data - M1)
    M2_Cluster = abs(Data - M2)
    Cluster = [C1.append(Data[i]) if M1_Cluster[i] < M2_Cluster[i] else C2.append(Data[i]) for i in (range(len(Data)))]
    M1 = sum(C1) / len(C1)
    M2 = sum(C2) / len(C2)
    if M1 == P1 and M2 == P2:
        break

print(M1, M2)
print(np.array(C1))
print(np.array(C2))