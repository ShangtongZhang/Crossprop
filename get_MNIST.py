import os

os.system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
os.system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
os.system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
os.system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
os.system('gunzip *.gz')
os.system('rm -rf *.gz')