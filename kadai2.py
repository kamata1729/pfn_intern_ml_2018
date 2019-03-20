from utils import *
from tqdm import tqdm


def read_img(path):
    with open(path) as f:
        f = f.read()
        f = f.split("\n")
    img = sum([vec.split() for vec in f[3:35]], [])
    img = cast(img, dtype=float)
    img = scalor_product(1/255, img)
    return img


def read_params(path="param.txt"):
    H = 256
    C = 23
    with open(path) as f:
        f = f.read()
        f = f.split("\n")
    result = {'W_1': [], 'b_1': [], 'W_2': [], 'b_2': [], 'W_3': [], 'b_3': []}

    for i in range(2*H + C + 3):
        if i < H:
            result['W_1'].append(cast(f[i].split(), dtype=float))
        if i == H:
            result['b_1'].append(cast(f[i].split(), dtype=float))
        if H < i and i < 2*H + 1:
            result['W_2'].append(cast(f[i].split(), dtype=float))
        if i == 2*H + 1:
            result['b_2'].append(cast(f[i].split(), dtype=float))
        if 2*H + 1 < i < 2*H + C + 2:
            result['W_3'].append(cast(f[i].split(), dtype=float))
        if i == 2*H + C + 2:
            result['b_3'].append(cast(f[i].split(), dtype=float))
    return result


def read_labels(filename="labels.txt"):
    with open(filename) as f:
        f = f.read()
        f = f.split("\n")
    return scalor_add(-1, cast(f[:-1], dtype=int))  # label range is 0 to 22


def predict(params, img):
    def linear(x, W, b):
        return add(mul(W, x), b)

    a_1 = linear(img, params['W_1'], params['b_1'][0])
    h_1 = relu(a_1)
    a_2 = linear(h_1, params['W_2'], params['b_2'][0])
    h_2 = relu(a_2)
    y = linear(h_2, params['W_3'], params['b_3'][0])
    result = sigmoid(y)
    return result


def predict_all():
    params = params = read_params()
    labels = read_labels()

    predicts = zero_vector(len(labels))
    for i in tqdm(range(len(labels))):
        img = read_img("pgm/{}.pgm".format(i+1))
        predicted_values = predict(params, img)
        predicts[i] = argmax(predicted_values)
    return accuracy(predicts, labels)

if __name__ == '__main__':
    acc = predict_all()
    print("accuracy: ", acc)

