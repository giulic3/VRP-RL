import tensorflow as tf
from misc_utils import debug_tensor


class Embedding(object):
    '''
    This class is the base class for embedding the input graph.
    '''
    def __init__(self, emb_type, embedding_dim):
        self.emb_type = emb_type  # linear
        self.embedding_dim = embedding_dim  # 128

    def __call__(self, input_pnt):
        # Returns the embedded tensor. Should be implemented in child classes
        pass


class LinearEmbedding(Embedding):
    '''
    This class implements linear embedding. It is only a mapping 
    to a higher dimensional space.
    '''
    def __init__(self, embedding_dim,_scope=''):
        '''
        Input:
            embedding_dim: embedding dimension
        '''
        super(LinearEmbedding, self).__init__('linear', embedding_dim)
        # 1D convolution layer
        # embedding_dim is the filters, in integer,
        # the dimensionality of the output space (i.e. the number of filters in the convolution).
        # 1 is the kernel_size
        # _scope belongs to the **kwargs
        self.project_emb = tf.layers.Conv1D(embedding_dim, 1, _scope=_scope+'Embedding/conv1d')

    def __call__(self, input_pnt):
        # emb_inp_pnt: [batch_size, max_time, embedding_dim]
        emb_inp_pnt = self.project_emb(input_pnt)
        # debug_tensor(emb_inp_pnt)
        # emb_inp_pnt = tf.Print(emb_inp_pnt,[emb_inp_pnt])
        return emb_inp_pnt


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    # random_uniform takes a Python array as the shape of the output tensor
    input_pnt = tf.random_uniform([2, 10, 2])
    # the embedding is created with size=128
    Embedding = LinearEmbedding(128)
    emb_inp_pnt = Embedding(input_pnt)
    sess.run(tf.global_variables_initializer())
    print(sess.run([emb_inp_pnt, tf.shape(emb_inp_pnt)]))
