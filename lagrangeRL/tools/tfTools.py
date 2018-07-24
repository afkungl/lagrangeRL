import tensorflow as tf


def tf_mat_vec_dot(matrix, vector):
    '''
    Matrix product between matrix and vector.
    '''
    return tf.matmul(matrix, tf.expand_dims(vector, 1))[:,0]

def tf_outer_product(first_vec, second_vec):
    '''
    Outer product of two vectors, outer(v,j)_ij = v[i]*v[j].
    '''
    return tf.einsum('i,j->ij', first_vec, second_vec)