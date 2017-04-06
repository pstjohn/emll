import theano
import theano.tensor as T
import numpy as np


class IsStable(theano.gof.Op):
    """
    Compute the eigenvalues and right eigenvectors of a square array.
    """

    _numop = staticmethod(np.linalg.eigvals)
    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 2
        w = theano.tensor.dscalar()
        return theano.gof.Apply(self, [x], [w])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w,) = outputs

        try:
            largest = np.max(np.real(np.asarray(self._numop(x))))

            if largest > 0:
                w[0] = np.array(-np.inf)
            else:
                w[0] = np.array(0.)
        except np.linalg.LinAlgError:
            w[0] = np.array(-np.inf)

    def infer_shape(self, node, shapes):
        return [()]

    def grad(self, inputs, g_outputs):
        return [T.zeros_like(inputs[0])]

isstable = IsStable()
