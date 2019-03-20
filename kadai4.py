import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import argparse


def cast(x: list, dtype: type):
    return list(map(lambda a: dtype(a), x))


def read_img(path):
    img = cv2.imread(path, 0) / 255
    return img.reshape(-1)


def read_params(path="param.txt"):
    H = 256
    C = 23
    N = 1024
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

    for key in result.keys():
        result[key] = np.array(result[key])
        if key.startswith('b'):
            result[key] = result[key][0]
    return result


def read_labels(filename="labels.txt"):
    with open(filename) as f:
        f = f.read()
        f = f.split("\n")
    return np.array(cast(f[:-1], dtype=int)) - 1  # label range is 0 to 22


def one_hot(t, length):
    return np.eye(length)[t]


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


def sign(x):
    x = x.copy()
    x[x > 0] = 1
    x[x <= 0] = -1
    return x


def predict(params, img):
    a_1 = params['W_1']@img + params['b_1']
    h_1 = relu(a_1)
    a_2 = params['W_2']@h_1 + params['b_2']
    h_2 = relu(a_2)
    y = params['W_3']@h_2 + params['b_3']
    f_x = softmax(y)
    return f_x


def predict_with_backward(params, img, t):
    def backward(p, q):
        return p * (q > 0)

    # forward
    a_1 = params['W_1']@img + params['b_1']
    h_1 = relu(a_1)
    a_2 = params['W_2']@h_1 + params['b_2']
    h_2 = relu(a_2)
    y = params['W_3']@h_2 + params['b_3']
    f_x = softmax(y)

    # backward
    nabla_y = - one_hot(t, len(f_x)) + f_x
    nabla_h2 = params['W_3'].T @ nabla_y
    nabla_a2 = backward(nabla_h2, a_2)
    nabla_h1 = params['W_2'].T @ nabla_a2
    nabla_a1 = backward(nabla_h1, a_1)
    nabla_x = params['W_1'].T @ nabla_a1

    return nabla_x


def baseline_img(img, eps_0=0.1):
    return img + eps_0 * (np.random.randint(0, 2, len(img))*2 - 1)


def fgsm_img(params, img, t, eps_0=0.1):
    nabla_x = predict_with_backward(params, img, t)
    img_fgsm = img + eps_0 * sign(nabla_x)
    return img_fgsm


#################################################
# Derection 1
#################################################


def execute_fgsm_repeat(params, labels, eps_0=0.01, repeat_count=0):
    predicts_fgsm = np.zeros(len(labels))
    predicts_baseline = np.zeros(len(labels))
    for i in tqdm(range(len(labels)), desc='eps_0={}'.format(eps_0)):
        img = read_img("pgm/{}.pgm".format(i+1))
        img_fgsm = fgsm_img(params, img, labels[i], eps_0)
        img_baseline = baseline_img(img, eps_0)
        for _ in range(repeat_count):
            img_fgsm = fgsm_img(params, img_fgsm, labels[i], eps_0)
            img_baseline = baseline_img(img_baseline, eps_0)
        predicts_fgsm[i] = predict(params, img_fgsm).argmax()
        predicts_baseline[i] = predict(params, img_baseline).argmax()

    return {"fgsm": np.sum(predicts_fgsm == labels) / len(labels),
            "baseline": np.sum(predicts_baseline == labels) / len(labels)}


#################################################
# Derection 2
#################################################


def fgsm_img_mono(params, img, t, rho=0.2):
    img_fgsm = img.copy()
    nabla_x = predict_with_backward(params, img, t)
    noise = np.zeros_like(nabla_x)
    noise[nabla_x > 0] = 1
    noise[nabla_x <= 0] = -1
    flag = np.argsort(np.abs(nabla_x)) > (1 - rho)*len(nabla_x)
    img_fgsm[flag] += noise[flag]
    img_fgsm = np.clip(img_fgsm, 0, 1)
    return img_fgsm


def baseline_img_mono(img, rho=0.2):
    noise = np.random.randint(-1, 2, len(img))
    flag = np.random.rand(len(img)) < rho
    return np.clip(img+flag*noise, 0, 1)


def execute_fgsm_mono(params, labels, rho=0.05):
    predicts_fgsm = np.zeros(len(labels))
    predicts_baseline = np.zeros(len(labels))
    for i in tqdm(range(len(labels)), desc='rho={}'.format(rho)):
        img = read_img("pgm/{}.pgm".format(i+1))
        img_fgsm_mono = fgsm_img_mono(params, img, labels[i], rho)
        img_baseline = baseline_img_mono(img, rho)
        predicts_fgsm[i] = predict(params, img_fgsm_mono).argmax()
        predicts_baseline[i] = predict(params, img_baseline).argmax()

    return {"fgsm": np.sum(predicts_fgsm == labels) / len(labels),
            "baseline": np.sum(predicts_baseline == labels) / len(labels)}


def plot(ygrid, ygrid_name, fgsm_accs, baseline_accs, filename):
    plt.plot(ygrid, fgsm_accs, label='FGSM')
    plt.plot(ygrid, baseline_accs, label='baseline')
    plt.xlabel(ygrid_name)
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(filename)
    print("saved: ", filename)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('direction', type=int, choices=[1,2,3])
    args = parser.parse_args()

    if args.direction == 1:
        params = read_params()
        labels = read_labels()
        max_count = 10
        fgsm_accs = np.zeros(max_count)
        baseline_accs = np.zeros(max_count)
        for i, repeat_count in enumerate(range(max_count)):
            result = execute_fgsm_repeat(params, labels, repeat_count=repeat_count)
            fgsm_accs[i] = result['fgsm']
            baseline_accs[i] = result['baseline']
        plot(list(range(max_count)), "repeat count", fgsm_accs, baseline_accs, "plot_repeat.png")

    if args.direction == 2:
        params = read_params()
        labels = read_labels()
        rhos = [0, 0.05, 0.1, 0.2, 0.5, 0.8]
        fgsm_accs = np.zeros_like(rhos)
        baseline_accs = np.zeros_like(rhos)
        for i, rho in enumerate(rhos):
            result = execute_fgsm_mono(params, labels, rho)
            fgsm_accs[i] = result['fgsm']
            baseline_accs[i] = result['baseline']
        plot(rhos, 'noise proportion', fgsm_accs, baseline_accs, 'plot_mono.png')

        




