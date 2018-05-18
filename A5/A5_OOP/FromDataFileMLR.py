from numpy import *
import csv

class FromDataFileMLR:
    def getTwoDecPoint(self, x):
        return float("%.2f" % x)

    # -------------------------------------------------------------------------------------------------

    def placeDataIntoArray(self, fileName):
        with open(fileName, mode='rbU') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
            dataArray = array([row for row in datareader], dtype=float64, order='C')

        if (min(dataArray.shape) == 1):  # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray

    # -------------------------------------------------------------------------------------------------

    def getAllOfTheData(self):
        TrainX = self.placeDataIntoArray('Train-Data.csv')
        TrainY = self.placeDataIntoArray('Train-pIC50.csv')
        ValidateX = self.placeDataIntoArray('Validation-Data.csv')
        ValidateY = self.placeDataIntoArray('Validation-pIC50.csv')
        TestX = self.placeDataIntoArray('Test-Data.csv')
        TestY = self.placeDataIntoArray('Test-pIC50.csv')
        return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY

    # -------------------------------------------------------------------------------------------------

    def rescaleTheData(self, TrainX, ValidateX, TestX):
        # 1 degree of freedom means (ddof) N-1 unbiased estimation
        TrainXVar = TrainX.var(axis=0, ddof=1)
        TrainXMean = TrainX.mean(axis=0)

        for i in range(0, TrainX.shape[0]):
            TrainX[i, :] = (TrainX[i, :] - TrainXMean) / sqrt(TrainXVar)
        for i in range(0, ValidateX.shape[0]):
            ValidateX[i, :] = (ValidateX[i, :] - TrainXMean) / sqrt(TrainXVar)
        for i in range(0, TestX.shape[0]):
            TestX[i, :] = (TestX[i, :] - TrainXMean) / sqrt(TrainXVar)

        return TrainX, ValidateX, TestX