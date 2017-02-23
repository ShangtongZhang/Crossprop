import numpy as np
import matplotlib.pyplot as plt
from Activation import *
from BackpropLearner import *
from CrosspropLearner import *

lr = 0.1
dims = [2, 2]
bias = [False, False]
act = 'sigmoid'
bp = BackpropLearner(lr, dims=list(dims), bias=bias, activation=act)
cp = CrossPropLearner(lr, dims=list(dims), bias=bias, activation=act)
# U = np.array([[0.6, 0.9]]).T
# W = np.array([100.0, 100.0])
ds = (dims[0] + int(bias[0]), dims[1] + int(bias[1]))
U = np.random.randn(*ds)
# U[:, 0] = U[:, 1]
W = np.random.randn(ds[1])
bp.U = np.copy(U)
bp.W = np.copy(W)
cp.U = np.copy(U)
cp.W = np.copy(W)
learners = [bp, cp]
labels = ['bp', 'cp']

# X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Y = np.array([0, 1, 0, 1])
Y = np.array([0, 1, 1, 0])
# X = np.array([[0]])
# Y = np.array([0])
if bias[0]:
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

epochs = 2000
errors = np.zeros((len(labels), epochs))
phi1 = np.zeros((len(labels), epochs))
phi2 = np.zeros((len(labels), epochs))
ws = np.zeros((len(labels), epochs))
bs = np.zeros((len(labels), epochs))

# print cp.W
# print cp.U
print bp.W
print bp.U

for ind, label in enumerate(labels):
    for ep in range(epochs):
        for i in range(X.shape[0]):
            learners[ind].predict(X[i, :])
            errors[ind, ep] += learners[ind].learn(Y[i])
        phi1[ind, ep] = learners[ind].phi[0]
        phi2[ind, ep] = learners[ind].phi[1]
        ws[ind, ep] = learners[ind].W[0]
        bs[ind, ep] = learners[ind].W[1]

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
    plt.plot(np.arange(epochs), phi1[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('Activation1')
plt.ylim([0, 1])
plt.legend()

plt.figure(2)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), phi2[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('Activation2')
plt.ylim([0, 1])
plt.legend()

plt.figure(3)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), ws[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('w')
plt.legend()

plt.figure(4)
for ind, label in enumerate(labels):
    plt.plot(np.arange(epochs), bs[ind, :], label=label)
plt.xlabel('Sweep')
plt.ylabel('bias')
plt.legend()

plt.show()
