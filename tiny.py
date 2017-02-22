import numpy as np
import matplotlib.pyplot as plt
from Activation import *

actFn = sigmoid
gradientActFn = gradientSigmoid

class BackpropLearner:
    def __init__(self, lr, U, w=1.0):
        self.lr = lr
        self.U = np.copy(U)
        self.w = w

    def predict(self, X):
        self.X = X
        self.net = np.dot(X, self.U)
        self.phi = actFn(self.net)
        self.y = self.w * self.phi
        return self.y

    def learn(self, target):
        error = target - self.y
        gradientW = -error * self.phi
        gradientU = -error * self.w * np.multiply(
            np.repeat(gradientActFn(self.phi, self.net), 2),
            self.X
        )
        # self.w -= self.lr * gradientW
        self.U -= self.lr * gradientU
        return 0.5 * error * error

class CrosspropLearner:
    def __init__(self, lr, U, w=1.0):
        self.lr = lr
        self.U = np.copy(U)
        self.w = w
        self.H = np.zeros(U.shape)

    def predict(self, X):
        self.X = X
        self.net = np.dot(X, self.U)
        self.phi = actFn(self.net)
        self.y = self.w * self.phi
        return self.y

    def learn(self, target):
        error = target - self.y
        gradientW = -error * self.phi
        gradientU = -error * self.w * np.multiply(
            np.repeat(self.phi, 2),
            self.H
        )
        self.H = np.multiply(self.H, 1 - self.lr * np.power(np.repeat(self.phi, 2), 2)) + \
            self.lr * error * np.repeat(gradientActFn(self.phi, self.net), 2)
        # self.w -= self.lr * gradientW
        self.U -= self.lr * gradientU
        # print error, gradientU, self.U, self.lr * error * np.repeat(gradientActFn(self.phi, self.net), 2)
        return 0.5 * error * error

lr = 0.15
U = np.array([0.6, 0.9])
bp = BackpropLearner(lr, U)
cp = CrosspropLearner(lr, U)
learners = [bp, cp]
labels = ['bp', 'cp']

X = 1.0
Y = 0.0
X = np.array([X, 1.0])

epochs = 200
errors = np.zeros((len(labels), epochs))
phi = np.zeros((len(labels), epochs))
ws = np.zeros((len(labels), epochs))

for ind, label in enumerate(labels):
    for ep in range(epochs):
        learners[ind].predict(X)
        phi[ind, ep] = learners[ind].phi
        ws[ind, ep] = learners[ind].w
        errors[ind, ep] = learners[ind].learn(Y)

print bp.w, bp.U
print cp.w, cp.U

plt.figure(0)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), errors[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('Error')

plt.figure(1)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), phi[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('Activation')

plt.figure(2)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), ws[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('w')

plt.legend()
plt.show()
