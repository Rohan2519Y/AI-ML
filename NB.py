import pandas as pd
import numpy as np
# 1 - Find out two classes
#   - Yes or No
# 2 - Create Data for Yes or No
# 3 - Find out Yes Feature and No Feature
# 
# 
# 
# 
# 
# 
# 
# 

class NB:
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.classes = list(set(Y))
        # print(self.classes)
        self.data = {}
        for i in self.classes:
            L = []
            for o, r in enumerate(self.Y):
                if i == r:
                    L.append(X[o])
            self.data[i] = L
        # print(self.data)

        self.prob = {}
        for i in self.classes:
            self.prob[i] = len(self.data[i]) / len(Y)
        # print(self.prob)
        print(np.array(self.data))

    # def predict(self, X):
    #     self.


X = [
    ['Sunny', 'Hot', 'Weak'],
    ['Sunny', 'Hot', 'Strong'],
    ['Rainy', 'Cool', 'Weak'],
    ['Rainy', 'Cool', 'Strong'],
    ['Sunny', 'Cool', 'Weak'],
    ['Rainy', 'Hot', 'Weak'],
]

Y = ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes']

model = NB()
model.fit(X, Y)