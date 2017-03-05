import numpy as np
import cPickle
import re
import os

n_classes=10
image_width=32
image_height=32
image_depth=3

# regular expression that matches a datafile
r_data_file = re.compile('^data_batch_\d+')

# training and validate datasets as numpy n-d arrays,
# apropriate portions of which are ready to be fed to the placeholder variables
train_all = {'data': [], 'labels': []}
validate_all = {'data': [], 'labels': []}
test_all = {'data': {}, 'labels': []}
label_names_for_validation_and_test = None


def unpickle(relpath):
    with open(relpath, 'rb') as fp:
        d = cPickle.load(fp)
    return d

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def prepare_input(data=None, labels=None):
    global image_height, image_width, image_depth
    assert (data.shape[1] == image_height * image_width * image_depth)
    assert (data.shape[0] == labels.shape[0])
    data *= 1.0 / np.max(data)
    labels = dense_to_one_hot(labels)
    # do mean normaization across all samples
    data = data.reshape([-1, image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)
    return data, labels

def load_cifar10(dataset_dir=None):
    global train_all, validate_all, label_names_for_validation_and_test
    trn_all_data = []
    trn_all_labels = []
    # for loading train dataset, iterate through the directory to get matchig data file
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            m = r_data_file.match(f)
            if m:
                relpath = os.path.join(root, f)
                d = unpickle(os.path.join(root, f))
                trn_all_data.append(d['data'])
                trn_all_labels.append(d['labels'])
    # concatenate all the  data in various files into one ndarray of shape
    # data.shape == (no_of_samples, 3072), where 3072=image_depth x image_height x image_width
    # labels.shape== (no_of_samples)
    train_x, train_y = (np.concatenate(trn_all_data).astype(np.float32),
                                    np.concatenate(trn_all_labels).astype(np.int32)
                                    )

    # load the only test data set for validation and testing
    # use only the first n_validate_samples samples for validating
    test_all = unpickle(os.path.join(dataset_dir, 'test_batch'))
    train_x, train_y = prepare_input(train_x, train_y)
    test_x, test_y = prepare_input(np.asarray(test_all['data'], dtype='float32'),
                                   np.asarray(test_all['labels']))
    return train_x, train_y, test_x, test_y

