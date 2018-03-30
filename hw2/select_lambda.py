import numpy as np 
import matplotlib.pyplot as plt 
import sys 

from src.util import *

lmbda_list = np.logspace(-2, -1, 20)

def parse_argv(argv):
    if len(argv) == 4:
        return argv[1], argv[2], argv[3]
    return './dataset/train_X.csv', './dataset/train_Y.csv', './experiments/lmbda.png'

if __name__ == '__main__':
    X_file, y_file, output_file = parse_argv(sys.argv)
    X, y = read_training_data(X_file, y_file)
    theta_init = initialize_theta(X, y)

    train_error_avg, val_error_avg = choose_parameters(lmbda_list, theta_init, h, J, X, y, eta=0.1, epoch=1500, n_fold=3, n_minibatch=1)

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.plot(lmbda_list, train_error_avg)
    ax.plot(lmbda_list, val_error_avg)
    plt.savefig(output_file)
