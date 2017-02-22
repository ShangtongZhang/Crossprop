import numpy as np
import matplotlib.pyplot as plt
from Activation import *
from BackpropLearner import *
from CrosspropLearner import *

lr = 0.15
dims = [2, 1]
bias = [False, False]
act = 'sigmoid'
bp = BackpropLearner(lr, dims=list(dims), bias=bias, activation=act)
cp = CrossPropLearner(lr, dims=list(dims), bias=bias, activation=act)
# U = np.array([[0.6, 0.9]]).T
# W = np.array([1.0])
# bp.U = np.copy(U)
# bp.W = np.copy(W)
# cp.U = np.copy(U)
cp.U[0] = 0
# cp.W = np.copy(W)
learners = [bp, cp]
labels = ['bp', 'cp']

# X = np.array([[0, 1], [2, 30]])
X = np.array([[0, 0], [2, 0]])
Y = np.array([0, 2])
if bias[0]:
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

epochs = 200
errors = np.zeros((len(labels), epochs))
phi = np.zeros((len(labels), epochs))
ws = np.zeros((len(labels), epochs))

# print cp.W
# print cp.U
print bp.W
print bp.U

for ind, label in enumerate(labels):
    for ep in range(epochs):
        for i in range(X.shape[0]):
            learners[ind].predict(X[i, :])
            phi[ind, ep] = learners[ind].phi
            ws[ind, ep] = learners[ind].W[0]
            errors[ind, ep] = learners[ind].learn(Y[i])

print bp.W
print bp.U
# print cp.W
# print cp.U
print '---'

plt.figure(0)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), errors[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('Error')
plt.legend()

plt.figure(1)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), phi[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('Activation')
plt.legend()

plt.figure(2)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), ws[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('w')
plt.legend()

plt.show()
