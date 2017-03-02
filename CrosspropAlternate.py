import numpy as np

class Crossprop_Alternate():
	def __init__(self, input_dim=20, hidden_dim=1000, output_dim=10, alpha_step_size=0.001, beta_step_size=0.001, lmbda=0.0, non_linearity='tanh'):
		# initialize the parameters of a neural network
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim  # regression task
		self.alpha_step_size = alpha_step_size
		self.beta_step_size = beta_step_size
		self.non_linearity = non_linearity
		self.lmbda = lmbda

		# init the weights
		self.hidden_output_weights = np.random.normal(0.0, 1.0,
													  size=(self.hidden_dim, self.output_dim))
		self.input_hidden_weights = np.random.normal(0.0, 1.0,
													  size=(self.input_dim, self.hidden_dim - 1))

		# self.hidden_output_weights = np.zeros((self.hidden_dim, self.output_dim))
		# self.input_hidden_weights = np.zeros((self.input_dim, self.hidden_dim - 1))
		# self.input_hidden_weights, _, _ = np.linalg.svd(np.random.random((self.input_dim, self.hidden_dim - 1)),
		# 												full_matrices=False)
		# _, _, self.hidden_output_weights = np.linalg.svd(np.random.random((self.hidden_dim, self.output_dim)),
		# 												full_matrices=False)
		# self.input_hidden_weights = self.input_hidden_weights.reshape((self.input_dim, self.hidden_dim - 1)) * np.sqrt(2.0)
		# self.hidden_output_weights = self.hidden_output_weights.reshape((self.hidden_dim, self.output_dim))
		self.input_hidden_traces = np.zeros((self.hidden_dim - 1, self.output_dim))

		# init variables that hold other relevant stuff for the neural net
		self.input_vector = np.zeros((self.input_dim, 1))
		self.features = np.zeros((self.hidden_dim, 1))
		# self.y = 0.0  # estimate computed by this neural net
		self.probability_estimates = np.zeros((self.output_dim, 1))
		return

	def apply_nonlinearity(self, pre_activations):
		if self.non_linearity == 'sigmoid':
			return 1.0 / (1 + np.exp(-pre_activations))
		elif self.non_linearity == 'tanh':
			return np.tanh(pre_activations)
		elif self.non_linearity == 'relu':
			return np.maximum(np.zeros(pre_activations.shape), pre_activations)

	def get_nonlinearity_derivative(self):
		if self.non_linearity == 'sigmoid':
			return self.features * (1 - self.features)
		elif self.non_linearity == 'tanh':
			return 1.0 - (self.features ** 2)
		elif self.non_linearity == 'relu':
			gradient_features = np.zeros(self.features.shape)
			gradient_features[self.features >= 0.0] = 1
			return gradient_features

	def compute_features(self):
		# input_vector = input_dim x 1
		# pre_activations = hidden_dim x 1
		pre_activations = np.dot(self.input_hidden_weights.transpose(), self.input_vector)
		self.features = np.ones((self.hidden_dim, 1))
		self.features[0 : self.hidden_dim - 1, :] = self.apply_nonlinearity(pre_activations)
		return self.features

	def set_input_vector(self, input_vector):
		self.input_vector = input_vector
		return

	def apply_softmax(self, pre_activations):
		e_x = np.exp(pre_activations - np.max(pre_activations))
		return e_x * 1.0 / e_x.sum()

	def make_estimate(self):
		self.compute_features()
		pre_activations = np.dot(self.hidden_output_weights.transpose(), self.features)
		self.probability_estimates = self.apply_softmax(pre_activations)
		return self.probability_estimates

	def compute_cross_entropy_error(self, true_label):
		return -np.multiply(true_label, np.log(self.probability_estimates)).sum()

	def get_softmax_grad(self, true_label):
		return (self.probability_estimates - true_label)

	def crosspropagate_errors(self, true_label):
		# hidden_output_weights_delta = hidden_dim x 1
		# input_hidden_weights_delta = input_dim x hidden_dim
		delta = self.get_softmax_grad(true_label)
		hidden_output_weights_delta = np.dot(self.features, delta.transpose())

		input_hidden_weights_interim = np.zeros((self.hidden_dim - 1, 1))
		for i in range(self.output_dim):
			# input_hidden_weights_interim += (delta[i] *
			# 								 np.multiply(self.features[0 : self.hidden_dim - 1, :], self.input_hidden_traces[:, i]))
			input_hidden_weights_interim[0 : self.hidden_dim - 1, 0] += (delta[i] * (self.features[0: self.hidden_dim - 1].flatten() * self.input_hidden_traces[:, i]))
			self.input_hidden_traces[:, i] = self.input_hidden_traces[:, i] * (1.0 - self.beta_step_size * (self.features[0 : self.hidden_dim - 1].flatten() ** 2)) +\
											 (self.beta_step_size * delta[i])
		input_hidden_weights_delta = np.dot(self.input_vector,
												   (input_hidden_weights_interim * self.get_nonlinearity_derivative()[0 : self.hidden_dim - 1, :]).transpose())

		self.input_hidden_weights = self.input_hidden_weights - (self.beta_step_size * input_hidden_weights_delta)
		self.hidden_output_weights = self.hidden_output_weights - (self.alpha_step_size * hidden_output_weights_delta)
		return