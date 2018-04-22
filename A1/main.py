"""
Group 8: Chris Banci / Esmond Liang
CS512 - Data Mining
2/2/18

Assignment 1
"""
import numpy as np


def GetDataFromDataFile():
    data = np.genfromtxt('data.txt', dtype='int')
    return data


def GetThreeRandomNumber(matrix, column1=-1, column2=-1, column3=-1):
    # set of available columns to select random columns from
    column_set = np.arange(matrix.shape[1])

    # if optional column parameters are entered, remove those columns from column_set
    if np.array_equal([column1, column2, column3], [-1, -1, -1]) is False:
        column_set = np.delete(column_set, [column1, column2, column3])

    # randomly selecting three unique column numbers
    random_set = np.random.choice(column_set, 3, replace=False)

    return random_set[0], random_set[1], random_set[2]


def MakeMatrix(matrix, column1, column2, column3):
    # splice random columns from matrix
    random1 = matrix[:, column1]
    random2 = matrix[:, column2]
    random3 = matrix[:, column3]

    # combine the random columns into a new matrix
    new_matrix = np.vstack((random1, random2, random3))
    new_matrix = np.transpose(new_matrix)
    return new_matrix


def AddingMatrices(matrix1, matrix2):
    result = np.add(matrix1, matrix2)
    return result


def AddingContentOfEachRow(matrix):
    result = np.sum(matrix, axis=1)
    result = np.matrix(result)
    result = np.transpose(result)
    return result


def Sorting(matrix, order='asc'):
    if order == 'desc':
        result = np.sort(matrix, axis=0)[::-1]
    elif order == 'asc':
        result = np.sort(matrix, axis=0)

    return result


def PrintOutput(originalMatrix, matrix1PreSort, matrix1, matrix2PreSort, matrix2, matrix3, matrix4, matrix5):
    print("The original matrix is as follows:")
    print(originalMatrix)
    print("\nChoosing 3 random columns to make Matrix 1:")
    print(matrix1PreSort)
    print("\nAfter sorting Matrix 1 in ascending order:")
    print(matrix1)
    print("\nChoosing 3 new random columns to make Matrix 2:")
    print(matrix2PreSort)
    print("\nAfter sorting Matrix 2 in descending order:")
    print(matrix2)
    print("\nAdding Matrix 1 and Matrix 2 into a 10x3 Matrix called Matrix 3:")
    print(matrix3)
    print("\nAdding the rows of Matrix 3 and putting it into a new 10x1 matrix called Matrix 4:")
    print(matrix4)
    print("\nAfter sorting matrix 4 in ascending order you get:")
    print(matrix5)


OriginalMatrix = GetDataFromDataFile()
#print(OriginalMatrix)

ColA, ColB, ColC = GetThreeRandomNumber(OriginalMatrix)
Matrix1 = MakeMatrix(OriginalMatrix, ColA, ColB, ColC)
Matrix1Presort = Matrix1
Matrix1 = Sorting(Matrix1, order='asc')
#print(Matrix1)

ColD, ColE, ColF = GetThreeRandomNumber(OriginalMatrix, ColA, ColB, ColC)
Matrix2 = MakeMatrix(OriginalMatrix, ColD, ColE, ColF)
Matrix2Presort = Matrix2
Matrix2 = Sorting(Matrix2, order='desc')
#print(Matrix2)

Matrix3 = AddingMatrices(Matrix1, Matrix2)
#print(Matrix3)

Matrix4 = AddingContentOfEachRow(Matrix3)
#print(Matrix4)

Matrix5 = Sorting(Matrix4, order='asc')
#print(Matrix5)
PrintOutput(OriginalMatrix, Matrix1Presort, Matrix1, Matrix2Presort, Matrix2, Matrix3, Matrix4, Matrix5)

