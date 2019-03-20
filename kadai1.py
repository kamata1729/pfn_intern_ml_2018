import math


def add(x: list, y: list):
    assert len(x) == len(y), 'lengths of both vectors must be same'
    return list(map(lambda a, b: a+b, x, y))


def sub(x: list, y: list):
    assert len(x) == len(y), 'lengths of both vectors must be same'
    return list(map(lambda a, b: a-b, x, y))


def product(x: list, y: list):
    assert len(x) == len(y), 'lengths of both vectors must be same'
    return sum(list(map(lambda x, y: x*y, x, y)))


def mul(A: list, x: list):
    assert len(A[0]) == len(x), 'incorrect dimension'
    return list(map(lambda a: product(a, x), A))


def scalor_add(a, x :list):
    return list(map(lambda x: a + x, x))

    
def scalor_product(a, x: list):
    return list(map(lambda x: a*x, x))


def transpose(A: list):
    height = len(A)
    width = len(A[0])
    result = [[0] * height for _ in range(width)]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][i] = A[i][j]
    return result


def relu(x: list):
    return list(map(lambda x: max(x, 0), x))


def sigmoid(x: list):
    exp_list = list(map(lambda a: math.exp(a), x))
    return list(map(lambda a: a/sum(exp_list), exp_list))
