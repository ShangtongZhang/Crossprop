import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

epochs = 200
learning_rates = np.power(2.0, np.arange(-13, -8))

for lr in learning_rates:
    path = 'tmp/MNIST_MNIST_'+str(lr)+'.bin'
    if not os.path.isfile(path):
        continue
    fr = open(path, 'rb')
    data = pickle.load(fr)['stats']
    train_loss, train_acc, test_loss, test_acc = data
    fr.close()

    labels = ['bp', 'cp']
    for i, label in enumerate(labels):
        plt.plot(np.arange(epochs), test_acc[i, 0, :], label=label)
    plt.legend()
    # plt.ylim([0, 100])
    plt.show()