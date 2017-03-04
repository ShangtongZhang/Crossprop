import numpy as np
import re
import matplotlib.pyplot as plt

num_epochs = 500
# step_size = 1.0 / (2 ** 10)
step_size_list = np.power(2.0, np.arange(-16, 0))
# file_name = 'logs/mnist_complete_hdim_1024_batchsize_100_ss_' + str(step_size) + '.txt'
# plot_file_name = 'plots/mnist_complete_hdim_1024_batchsize_100_ss_' + str(step_size) + '.txt'

test_cp = np.zeros((num_epochs, 1))
test_bp = np.zeros((num_epochs, 1))
test_mat = [test_cp, test_bp]

def main(ss):
	# global file_name, plot_file_name
	file_name = 'logs_mnist_convnet/mnist_complete_hdim_1024_batchsize_100_ep_500_ss_' + str(ss) + '.txt'
	plot_file_name = 'plots/mnist_convnet_training_complete_hdim_1024_batchsize_100_ep_500_ss_' + str(ss) + '.png'

	f = open(file_name)
	contents = f.readlines()
	f.close()
	contents = [x.strip() for x in contents]
	i = 0
	mat_id = 0
	for line_i in contents:
		if line_i.startswith('INFO:root:training_loss'):
			if i == num_epochs:
				i = 0
				mat_id = 1
			test_mat[mat_id][i] = float(re.findall("\d+\.\d+", line_i)[0])
			i += 1

	plt.clf()
	x = np.arange(1, test_cp.shape[0] + 1)
	plt.errorbar(x, test_bp, label='backprop', linewidth=3.0)
	plt.errorbar(x, test_cp, label='crossprop', linewidth=3.0)
	diff = test_bp[-1] - test_cp[-1]
	plt.title('bp: {}; cp: {}; diff: {}'.format(test_bp[-1], test_cp[-1], diff))
	plt.xlabel('Sweeps')
	plt.ylabel('Training loss')
	plt.legend(loc='best')
	# plt.ylim([0.0, 1.0])
	plt.savefig(plot_file_name)

if __name__ == '__main__':
	for ss in step_size_list:
		main(ss)
