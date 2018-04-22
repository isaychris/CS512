# =====================================================================================================================
# Group 8: Chris Banci / Esmond Liang
# CS512 - Data Mining
# 2/26/18
#
# Assignment 2 - REDO
# =====================================================================================================================
import numpy as np

class SquareMatrix:
    out = open('output.txt', 'w')

    # Purpose: gets dimension size from user input.
    def getDimension(self):
        try:
            self.print('Enter the dimension of your matrix: ', esc='')
            N = int(input())
            self.out.write(str(N) + '\n')

            if N <= 3:
                self.print(' - [Error] This dimension is out of bounds. Exiting program ...')
                exit(1)
            elif N > 6:
                self.print(' - [Error] We can only handle up to 6 dimension at this time. Exiting program ...')
                exit(1)
            return N

        except ValueError:
            self.print(' - [Error] Invalid input. Exiting program ...')
            exit(1)

    # Purpose: loads the matrix from file, taking the values within N by N
    def getMatrix(self, N, file):
        data = np.genfromtxt(file, dtype='str')

        for x in np.nditer(data):
            if not str(x[...]).isdigit():
                self.print(' - [Error] The matrix loaded contains non-numeric values. Exiting program ...')
                exit(1)

        data = data.astype(np.int)

        if data.size < (N * N):
            self.print(' - [Error] There are less than ' + str(N * N) + ' numbers in the file. Exiting program ...')
            exit(1)

        data = data[:N, :N]
        return data

    # Purpose: multiples each number in m1 with each number in m2
    def product(self, M1, M2):
        result = np.multiply(M1, M2)
        return result

    # Purpose: returns the dot product of two matrices
    def dotProduct(self, M1, M2):
        result = np.dot(M1, M2)
        return result

    # Purpose: shifts the orientation of the matrix
    def transpose(self, M):
        result = np.transpose(M)
        return result

    # Purpose: divides each number in m1 by each number in m2
    def divide(self, M1, M2):
        result = np.zeros(M2.shape)
        np.divide(M1, M2, out=result, where=(M2 != 0))

        if 0 in M2:
            result = result.astype(np.object_)
            result[result == 0] = 'undefined'
        return result

    # Purpose: custom print function that prints output into console and file.
    def print(self, string, esc='\n'):
        print(string, end=esc)
        self.out.write(string + esc)


def main():
    myMatrix = SquareMatrix()
    N = myMatrix.getDimension()
    M1 = myMatrix.getMatrix(N, 'file1.txt')
    M2 = myMatrix.getMatrix(N, 'file2.txt')
    M1_Multiply_M2 = myMatrix.product(M1, M2)
    M1_DotMultiply_M2 = myMatrix.dotProduct(M1, M2)
    M1_Trans = myMatrix.transpose(M1)
    M2_Trans = myMatrix.transpose(M2)
    M1Trans_Multiply_M2Trans = myMatrix.product(M1_Trans, M2_Trans)
    M1Trans_DotMultiply_M2Trans = myMatrix.dotProduct(M1_Trans, M2_Trans)
    M1_Divide_M2 = myMatrix.divide(M1, M2)

    myMatrix.print('=============================================================')
    myMatrix.print('The content of the first matrix is:')
    myMatrix.print(str(M1))

    myMatrix.print('=============================================================')
    myMatrix.print('The content of the second matrix is:')
    myMatrix.print(str(M2))

    myMatrix.print('=============================================================')
    myMatrix.print('The product of the two matrices is:')
    myMatrix.print(str(M1_Multiply_M2))

    myMatrix.print('=============================================================')
    myMatrix.print('The dot-product of the two matrices is:')
    myMatrix.print(str(M1_DotMultiply_M2))

    myMatrix.print('=============================================================')
    myMatrix.print('The transpose of the first matrix is:')
    myMatrix.print(str(M1_Trans))

    myMatrix.print('=============================================================')
    myMatrix.print('The transpose of the second matrix is:')
    myMatrix.print(str(M2_Trans))

    myMatrix.print('=============================================================')
    myMatrix.print('The product of the transpose of the two matrices is:')
    myMatrix.print(str(M1Trans_Multiply_M2Trans))

    myMatrix.print('=============================================================')
    myMatrix.print('The dot product of the transpose of the two matrices is:')
    myMatrix.print(str(M1Trans_DotMultiply_M2Trans))

    myMatrix.print('=============================================================')
    myMatrix.print('The result of matrix1 divided by matrix2 is:')
    myMatrix.print(str(M1_Divide_M2))
    myMatrix.print('=============================================================')


if __name__ == "__main__":
    main()

# =====================================================================================================================