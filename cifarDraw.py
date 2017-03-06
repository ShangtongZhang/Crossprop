import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

epochs = 200
learning_rates = np.power(2.0, np.arange(-13, -8))

for lr in learning_rates:
    fr = open('tmp/cifar10_AllCNN_'+str(lr)+'.bin', 'rb')
    train_loss, test_loss, test_acc = pickle.load(fr)['stats']
    fr.close()

    labels = ['cp', 'bp']
    for i, label in enumerate(labels):
        plt.plot(np.arange(epochs), train_loss[i, :], label=label)
    plt.legend()
    plt.ylim([0, 100])
    plt.show()