import numpy as np
import math
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, X, Y):
        self.__X = X
        self.__Y = Y

    def CC(self):
        Xsum = np.sum(self.__X)
        Ysum = np.sum(self.__Y)
        Xsqr = np.sum(self.__X * self.__X)
        Ysqr = np.sum(self.__Y * self.__Y)
        XY = np.sum(self.__X * self.__Y)
        R = (len(self.__X) * XY - (Xsum * Ysum))/ (math.sqrt((len(self.__X) * Xsqr - (Xsum * Xsum)) * (len(self.__X) * Ysqr - (Ysum * Ysum))))
        print("Corelation : ", R)

    def slope(self):
        self.__Xmean = np.mean(self.__X)
        self.__Ymean = np.mean(self.__Y)
        self.__b1 = np.sum((self.__X - self.__Xmean) * (self.__Y - self.__Ymean)) / np.sum((self.__X - self.__Xmean) * (self.__X - self.__Xmean))
        print("Slope : ", self.__b1)

    def intersept(self):
        self.__b0 = self.__Ymean - self.__b1 * self.__Xmean
        print("Intersept : ", self.__b0)

    def Predict(self, n):
        self.inp_arr = np.array(n)
        self.new_pred = self.__b0 + (self.__b1 * self.inp_arr)

    def MSE(self):
        self.diff = self.__Y - self.new_pred
        plt.scatter(self.__X, self.diff)
        plt.axhline(0)
        plt.show()
        sq = self.diff ** 2
        mse = sq.sum()
        N = len(self.__X)
        return mse / N

    def drawRegression(self):
        plt.scatter(self.__X, self.__Y)
        line_x = np.array([self.__X.min(), self.__X.max()])
        line_y = self.__b0 + (self.__b1 * line_x)
        plt.plot(line_x, line_y, color='red')
        plt.xlabel("Amount")
        plt.ylabel("Tip")
        plt.show()