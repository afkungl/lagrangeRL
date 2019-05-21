import tensorflow as tf
import numpy as np

def tf_mat_vec_dot(matrix, vector):
    '''
    Matrix product between matrix and vector.
    '''
    return tf.matmul(matrix, tf.expand_dims(vector, 1))[:, 0]

def leaky_relu(x):
    '''
    Leaky relus does not exist in tensorflow 1.0
    '''

    return tf.nn.relu(x) - 0.1 * tf.nn.relu(-x)

# Further helper functions to calculate the Jacobian

def map(f, x, dtype=None, parallel_iterations=10):
    '''
    Apply f to each of the elements in x using the specified number of parallel iterations.

    Important points:
    1. By "elements in x", we mean that we will be applying f to x[0],...x[tf.shape(x)[0]-1].
    2. The output size of f(x[i]) can be arbitrary. However, if the dtype of that output
       is different than the dtype of x, then you need to specify that as an additional argument.
    '''
    if dtype is None:
        dtype = x.dtype

    n = tf.shape(x)[0]
    loop_vars = [
        tf.constant(0, n.dtype),
        tf.TensorArray(dtype, size=n),
    ]
    _, fx = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1, result.write(j, f(x[j]))),
        loop_vars,
        parallel_iterations=parallel_iterations
    )
    return fx.stack()

def jacobian(fx, x, parallel_iterations=10):
    '''
    Given a tensor fx, which is a function of x, vectorize fx (via tf.reshape(fx, [-1])),
    and then compute the jacobian of each entry of fx with respect to x.
    Specifically, if x has shape (m,n,...,p), and fx has L entries (tf.size(fx)=L), then
    the output will be (L,m,n,...,p), where output[i] will be (m,n,...,p), with each entry denoting the
    gradient of output[i] wrt the corresponding element of x.
    '''
    return map(lambda fxi: tf.gradients(fxi, x)[0],
               tf.reshape(fx, [-1]),
               dtype=x.dtype,
               parallel_iterations=parallel_iterations)

def indicator_func(x, low, upper):
    '''
        The indicator function in tensorflow
        The value is one exactly between low and upper, otherwise zero
    '''

    return 0.5 * (tf.sign(upper - x) + 1.0) * 0.5 * (tf.sign(x - low) + 1.)

def homFunc(x, low, upper, width):
    '''
        Auxiliary function for the nudging modulated homeostasis.

        Args:
            x: input value
            low: lower limit
            upper: upper limit
            width: width of the transition ot zero

        Return:
             x < low - width: 0.0
             low - width < x < low: sinusoidal ramp up
             low < x < upper: 1.0
             upper < x < upper + width: sinusoidal ramp down
             upper + width < x: 0.0
    '''

    return 0.5 * (1. - tf.cos((x - (low - width))/width * np.pi)) * indicator_func(x, low - width, low) \
        + indicator_func(x, low, upper) \
        + 0.5 * (1. + tf.cos((x - upper)/width * np.pi)) * \
        indicator_func(x, upper, upper + width)
