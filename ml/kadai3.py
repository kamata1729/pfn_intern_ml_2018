import os
import random
import argparse
from tqdm import tqdm

from utils import *
from kadai2 import *


def predict_with_backward(params, img, t):
    def linear(x, W, b):
        return add(mul(W, x), b)

    def backward(x, y):
        assert len(x) == len(y), 'lengths of both vectors must be same'
        return list(map(lambda p, q: p if q > 0 else 0, x, y))

    # forward
    a_1 = linear(img, params['W_1'], params['b_1'][0])
    h_1 = relu(a_1)
    a_2 = linear(h_1, params['W_2'], params['b_2'][0])
    h_2 = relu(a_2)
    y = linear(h_2, params['W_3'], params['b_3'][0])
    f_x = softmax(y)

    # backward
    nabla_y = sub(f_x, one_hot(t, len(f_x)))
    nabla_h2 = mul(transpose(params['W_3']), nabla_y)
    nabla_a2 = backward(nabla_h2, a_2)
    nabla_h1 = mul(transpose(params['W_2']), nabla_a2)
    nabla_a1 = backward(nabla_h1, a_1)
    nabla_x = mul(transpose(params['W_1']), nabla_a1)

    return nabla_x


def fgsm_img(params, img, t, eps_0=0.1):
    nabla_x = predict_with_backward(params, img, t)
    img_fgsm = add(img, scalor_product(eps_0, sign(nabla_x)))
    return img_fgsm


def encode_pgm(img, path):
    folder_path = ''.join(path.split('/')[:-1])
    os.makedirs(folder_path, exist_ok=True)
    img = scalor_product(1/max(img), img)
    img = scalor_product(255, img)
    img = list(map(lambda x: max(int(x), 0), img))
    des = "P2\n32 32\n255\n"
    for i in range(32):
        des += ' '.join(cast(img[i*32: (i+1)*32], dtype=str)) + '\n'
    with open(path, mode='w') as f:
        f.write(des)


def baseline_img(img, eps_0=0.1):
    eps = [eps_0*random.randint(-1, 1) for _ in range(len(img))]
    return add(img, eps)


def execute_fgsm(params, labels, eps_0=0.1, save_fgsm=False, save_baseline=False):
    predicts_fgsm = zero_vector(len(labels))
    predicts_baseline = zero_vector(len(labels))
    for i in tqdm(range(len(labels)), desc='eps_0={}'.format(eps_0)):
        img = read_img("pgm/{}.pgm".format(i+1))
        img_fgsm = fgsm_img(params, img, labels[i], eps_0)
        predicts_fgsm[i] = argmax(predict(params, img_fgsm))
        if save_fgsm:
            encode_pgm(img_fgsm, 
            "pgm_fgsm_{}/{}.pgm".format(str(eps_0).replace('.', ''), i+1))

        img_baseline = baseline_img(img, eps_0)
        predicts_baseline[i] = argmax(predict(params, img_baseline))
        if save_baseline:
            encode_pgm(img_baseline, 
            "pgm_baseline_{}/{}.pgm".format(str(eps_0).replace('.', ''), i+1))

    return {"fgsm": accuracy(predicts_fgsm, labels),
            "baseline": accuracy(predicts_baseline, labels)}


def plot(epss, fgsm_accs, baseline_accs, filename='plot.png'):
    import matplotlib.pyplot as plt
    plt.plot(epss, fgsm_accs, label='FGSM')
    plt.plot(epss, baseline_accs, label='baseline')
    plt.xlabel('eps_0')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(filename)
    print("saved ", filename)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps_0', type=float)
    parser.add_argument('--save_fgsm', action='store_true')
    parser.add_argument('--save_baseline', action='store_true')
    parser.add_argument('--show_graph', action='store_true')
    args = parser.parse_args()

    assert not args.eps_0 or not args.show_graph, "either --eps_0 or --show_graph can be specified"

    if args.eps_0 or (not args.eps_0 and not args.show_graph):
        eps_0 = args.eps_0 if args.eps_0 else 0.1
        params = read_params()
        labels = read_labels()
        result = execute_fgsm(params, labels, eps_0,
                              args.save_fgsm, args.save_baseline)
        print("accuracy fgsm (eps_0={}): {:.7f}".format(
            args.eps_0, result['fgsm']))
        print("accuracy baseline (eps_0={}): {:.7f}".format(
            args.eps_0, result['baseline']))

    if args.show_graph:
        epss = [0, 0.05, 0.1, 0.2, 0.5, 0.8]
        fgsm_accs = zero_vector(len(epss))
        baseline_accs = zero_vector(len(epss))

        params = read_params()
        labels = read_labels()

        for i in range(len(epss)):
            if epss[i] == 0:
                pred = predict_all()
                fgsm_accs[i] = pred
                baseline_accs[i] = pred
            else:
                result = execute_fgsm(params, labels, epss[i],
                                    args.save_fgsm, args.save_baseline)
                fgsm_accs[i] = result['fgsm']
                baseline_accs[i] = result['baseline']
            print("accuracy fgsm (eps_0={}): {:.7f}".format(
                epss[i], fgsm_accs[i]))
            print("accuracy baseline (eps_0={}): {:.7f}".format(
                epss[i], baseline_accs[i]))
        
        plot(epss, fgsm_accs, baseline_accs)
    

