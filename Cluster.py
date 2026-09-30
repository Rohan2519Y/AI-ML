import numpy as np
import math

class Kmean:

    def __init__(self, data, M1, M2):
        self.data = data
        self.M1 = M1
        self.M2 = M2
        
    def fit(self):
        M1 = self.M1
        M2 = self.M2
        M1_Cluster = np.array([])
        M2_Cluster = np.array([])
        while True:
            P1, P2 = M1, M2
            C1 = []
            C2 = []
            M1_Cluster = abs(Data - M1)
            M2_Cluster = abs(Data - M2)
            for i in (range(len(Data))):
                if M1_Cluster[i] < M2_Cluster[i] : C1.append(Data[i]) 
                else : C2.append(Data[i])
            M1 = sum(C1) / len(C1)
            M2 = sum(C2) / len(C2)
            if M1 == P1 and M2 == P2:
                break
        self.M1 = M1
        self.M2 = M2
        self.M1_Cluster = M1_Cluster
        self.M2_Cluster = M2_Cluster

    def show(self):
        print("Cluster : ", list(self.M1_Cluster + self.M2_Cluster))
        print("Centeroid : ", list(self.M1 + self.M2))



Data = np.array([2, 4, 10, 12, 3, 20, 30, 11, 25])
model = Kmean(Data, 4, 11)
print(model.fit())