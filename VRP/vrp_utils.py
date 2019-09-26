import numpy as np
import tensorflow as tf
import os
import warnings
import collections
from misc_utils import debug_tensor, tf_print


def create_VRP_dataset(
        n_problems,
        n_cust,
        data_dir,
        seed=None,
        data_type='train'):
    '''
    This function creates VRP instances and saves them on disk. If a file is already available,
    it will load the file.
    Input:
        n_problems: number of problems to generate (default is 1000)
        n_cust: number of customers in the problem (default is 10)
        data_dir: the directory to save or load the file (default is data/)
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        data: a numpy array with shape [n_problems x (n_cust+1) x 3]
        in the last dimension, we have x,y,demand for customers. The last node is for depot and
        it has demand 0.
        picture a cube where you have n_problems rows, n_cust+1 columns and the nodes positions and demands as depth
     '''

    # Set random number generator
    n_nodes = n_cust + 1
    if seed == None:
        rnd = np.random
    else:
        # container for the Mersenne Twister pseudo-random number generator
        rnd = np.random.RandomState(seed)

    # build task name and datafiles
    task_name = 'vrp-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes, data_type)
    fname = os.path.join(data_dir, task_name)

    # create/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname, delimiter=' ')
        data = data.reshape(-1, n_nodes, 3)
    else:
        # Generate a training set of size n_problems
        print('Creating dataset for {}...'.format(task_name))
        # Draw samples from a uniform distribution, params are (low, high, size)
        x = rnd.uniform(0, 1, size=(n_problems, n_nodes, 2))
        # randint(low[, high, size, dtype]) 	Return random integers from low (inclusive) to high (exclusive).
        d = rnd.randint(1, 10, [n_problems, n_nodes, 1])
        d[:, -1] = 0  # Demand of depot, it's the last element
        # Data is the concatenation of static and dynamic elements
        data = np.concatenate([x, d], 2)
        np.savetxt(fname, data.reshape(-1, n_nodes*3))

    return data


class DataGenerator(object):
    def __init__(self, args):
        '''
        This class generates VRP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes']: number of nodes. nodes = cust + 1,
                cause the last one is the depot node,
                args['n_cust']: number of customers
                args['batch_size']: batch size for training

        '''
        self.args = args
        # init the random number generator and set the seed
        self.rnd = np.random.RandomState(seed=args['random_seed'])
        print('Created train iterator.')

        # create test data
        self.n_problems = args['test_size']
        self.test_data = \
            create_VRP_dataset(
                self.n_problems,
                args['n_cust'],
                './data',
                seed=args['random_seed']+1,
                data_type='test')

        self.reset()

    def reset(self):
        self.count = 0
        
    # Differently from test dataset, the training dataset is created 'online'
    def get_train_next(self):
        '''
        Get next batch of problems for training
        Returns:
            input_data: data with shape [batch_size x max_time x 3]
        '''
        # uniform(low, high, size), high is excluded, size is the output shape
        # generate positions of points as couples in the grid [0...1, 0...1]
        input_pnt = self.rnd.uniform(0, 1, size=(self.args['batch_size'], self.args['n_nodes'], 2))
        # generate the nodes demands as random integer numbers from 1 to 10
        demand = self.rnd.randint(1, 10, [self.args['batch_size'], self.args['n_nodes']])
        demand[:, -1] = 0  # demand of depot
        # generate input_data as concatenation of nodes positions and respective demands
        input_data = np.concatenate([input_pnt, np.expand_dims(demand, 2)], 2)

        return input_data

# f used to loop through batches
    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
        # move forward in the test set
        if self.count < self.args['test_size']:
            input_pnt = self.test_data[self.count:self.count+1]
            self.count += 1
        else:
            warnings.warn("The test iterator reset.")
            self.count = 0
            input_pnt = self.test_data[self.count:self.count+1]
            self.count += 1

        return input_pnt

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data


# a State is defined by (load of the vehicle, demands, demand satisfied and mask for the nodes)
class State(collections.namedtuple("State", ("load", "demand", 'd_sat', "mask"))):
    pass


class Env(object):
    def __init__(self,
                 args):
        '''
        This is the environment for VRP.
        Inputs:
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 2
        '''
        self.capacity = args['capacity']  # Max capacity of the truck
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        # Insert a placeholder for a tensor that will always be fed
        # placeholder is simply a variable that we will assign data to at a later date (e.g. after graph evaluation)
        # it allows to create our operations and build our computation graph, without needing the data
        # input_data: (?, 11, 3) - > (n_problems x n_nodes x dim+1)
        self.input_data = tf.placeholder(tf.float32, shape=[None, self.n_nodes, self.input_dim])
        # input_pnt: (?, 11, 2)
        self.input_pnt = self.input_data[:, :, :2]
        # takes the first starting from right, so the last
        self.demand = self.input_data[:, :, -1]
        # batch_size is the number of problems considered at a time for training
        # where each problem instance is a sequence of nodes with positions and demands
        self.batch_size = tf.shape(self.input_pnt)[0]

    def reset(self, beam_width=1):
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''

        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width

        self.input_pnt = self.input_data[:, :, :2]
        self.demand = self.input_data[:, :, -1]

        # modify the self.input_pnt and self.demand for beam search decoder
        # self.input_pnt = tf.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width, 1])

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam])*self.capacity

        # create mask
        self.mask = tf.zeros([self.batch_size*beam_width, self.n_nodes],
                dtype=tf.float32)

        # update mask -- mask if customer demand is 0 and depot
        self.mask = tf.concat([
            tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
            tf.ones([self.batch_beam, 1])], 1)
        # return the reset state
        state = State(
            load=self.load,
            demand=self.demand,
            d_sat=tf.zeros([self.batch_beam, self.n_nodes]),  # batch_beam = batch_size when beam_width=1
            mask=self.mask)  # specify masking scheme for the nodes

        return state

    # Run one step of the environment and update demands, loads and masks
    def step(self,
             idx,
             beam_parent=None):

        # If the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = \
                tf.expand_dims(
                    tf.tile(tf.cast(tf.range(self.batch_size), tf.int64), [self.beam_width]),
                    1)
            # batchedBeamIdx: [batch_size*beam_width]
            batchedBeamIdx= batchBeamSeq + tf.cast(self.batch_size, tf.int64)*beam_parent
            # demand:[batch_size*beam_width x sourceL]
            self.demand = tf.gather_nd(self.demand, batchedBeamIdx)
            # load:[batch_size*beam_width]
            self.load = tf.gather_nd(self.load, batchedBeamIdx)
            # MASK: [batch_size x beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask, batchedBeamIdx)

        # batch_beam = batch_size x beam_width
        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
        # concat along column axis -> shape (batch_beam x 2), it's like adding one column
        # this is needed to give an index in the batch to each node
        batched_idx = tf.concat([BatchSequence, idx], 1)

        # How much of the demand is satisfied - takes minimum from demand and load
        # at the beginning, self.load = max capacity
        #  (it's a tensor of 128 entries since we're considering a batch of problems)
        # d_sat must contain at the end the total demand satisfied
        d_sat = tf.minimum(tf.gather_nd(self.demand, batched_idx), self.load)  # TODO??

        # Update the demand, insert values into the tensor
        # d_scatter is a Tensor how much was already satisfied (for each node)
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand), tf.int64))
        # Update new demands
        self.demand = tf.subtract(self.demand, d_scatter)

        # Update vehicle load subtracting total demand already satisfied
        self.load -= d_sat

        # Refill the truck -- idx: [10,9,10] -> load_flag: [1 0 1]
        load_flag = tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32), 1)
        self.load = tf.multiply(self.load, 1-load_flag) + load_flag * self.capacity

        # The mask is then used in the decoding step
        # Mask for customers with zero demand
        self.mask = tf.concat(
                # check which demands = 0 and take last column (that would be the demand)
                [tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                tf.zeros([self.batch_beam, 1])],
            1)

        # Mask if load = 0
        # Mask if in depot and there is still a demand
        self.mask += \
            tf.concat(
                [tf.tile(
                    tf.expand_dims(tf.cast(tf.equal(self.load, 0), tf.float32), 1),
                    [1, self.n_cust]),
                tf.expand_dims(
                    tf.multiply(
                        tf.cast(tf.greater(tf.reduce_sum(self.demand, 1), 0), tf.float32),
                        tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32))),
                    1)],
                1)
        # Returns the new state with updated load, demand, d_sat and mask
        state = State(
            load=self.load,
            demand=self.demand,
            d_sat=d_sat,
            mask=self.mask)

        return state


# definition of reward_func for VRP problem
# used in the build_model() function to compute the reward
def reward_func(sample_solution):
    """The reward for the VRP task is defined as the
    negative value of the route length (this way it must be maximized)

    Args:
        sample_solution : a list tensor of length decode_len (there are 16 elements)
        # where each of them has shape [batch_size x input_dim]
        (this arg is not used) demands satisfied: a list tensor of size decode_len of shape [batch_size]
        decode_len: (max) length of the solution/ max length of the sequence produced by the decoder
        batch_size: number of nodes (or problems???) considered in the solution ???
        input_dim: 2 (couple of values for node position)
    Returns:
        rewards: tensor of size [batch_size]

    Example:
        sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
        sourceL = 3 (the depot?)
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                                    #  [6,6]]
                                                    # [[1,1]
                                                    #  [2,2]]
                                                    # [[3,3]
                                                    #  [4,4]] ]
    """
    # Make sample_solution of shape [sourceL x batch_size x input_dim]
    sample_solution = tf.stack(sample_solution, 0)
    # Tilting of solution is needed to compute then the distance between pairs of following nodes
    sample_solution_tilted = \
        tf.concat((
            tf.expand_dims(sample_solution[-1], 0),
            sample_solution[:-1]),
        0)
    # Get the reward based on the route lengths
    # This computes the Euclidean distance
    route_lens_decoded = tf.reduce_sum(
        tf.pow(
            tf.reduce_sum(
                tf.pow((sample_solution_tilted - sample_solution), 2)
                , 2),
            .5),
        0)
    return route_lens_decoded
