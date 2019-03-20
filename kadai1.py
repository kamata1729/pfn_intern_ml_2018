from utils import *

if __name__ == '__main__':
    x = [-1,2]
    y = [2,3]
    A = [[1, 2], [3,4]]
    print("x = ", x)
    print("y = ", y)
    print("A = ", A)
    print("x+y = ", add(x, y))
    print("Ax = ", mul(A, x))
    print("A^T = ", transpose(A))
    print("reu(x) = ", relu(x))
    print("softmax(x) = ", softmax(x))