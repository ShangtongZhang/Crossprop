import numpy as np
import pickle

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs

fr = open('YAD.bin', 'rb')
totalTrainX, totalTestX, totalTrainY, totalTestY = pickle.load(fr)
fr.close()

print totalTrainX.shape
totalTrainX, evals, evecs = PCA(totalTrainX, 20)
print totalTrainX.shape
print totalTestX.shape
totalTestX, evals, evecs = PCA(totalTestX, 20)
print totalTestX.shape
print evals

fw = open('YADReduced.bin', 'wb')
pickle.dump([totalTrainX, totalTestX, totalTrainY, totalTestY], fw)
fw.close()

