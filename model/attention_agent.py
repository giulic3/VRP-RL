import tensorflow as tf
import numpy as np
import time
from shared.embeddings import LinearEmbedding
from shared.decode_step import RNNDecodeStep
from misc_utils import debug_tensor


class RLAgent(object):

    def __init__(self,
                args,
                prt,
                env,
                dataGen,
                reward_func,
                clAttentionActor,
                clAttentionCritic,
                is_train=True,
                _scope=''):
        '''
        This class builds the model and runs test and train.
        Inputs:
            args: arguments. See the description in config.py file.
            prt: print controller which writes logs to a file.
            env: an instance of the environment.
            dataGen: a data generator which generates data for test and training.
            reward_func: the function which is used for computing the reward. In the
                        case of TSP and VRP, it returns the tour length.
            clAttentionActor: Attention mechanism that is used in actor.
            clAttentionCritic: Attention mechanism that is used in critic.
            is_train: if true, the agent is used for training; else, it is used only
                        for inference.
        '''

        self.args = args
        self.prt = prt
        self.env = env
        self.dataGen = dataGen
        self.reward_func = reward_func
        self.clAttentionCritic = clAttentionCritic
        # a LinearEmbedding is simply a 1D convolution
        self.embedding = LinearEmbedding(args['embedding_dim'], _scope=_scope+'Actor/')
        self.decodeStep = RNNDecodeStep(clAttentionActor,
                        args['hidden_dim'],
                        use_tanh=args['use_tanh'],
                        tanh_exploration=args['tanh_exploration'],
                        n_glimpses=args['n_glimpses'],
                        mask_glimpses=args['mask_glimpses'],
                        mask_pointer=args['mask_pointer'],
                        forget_bias=args['forget_bias'],
                        rnn_layers=args['rnn_layers'],
                        _scope='Actor/')
        # Get an existing variable with these parameters or create a new one
        # It's trainable
        self.decoder_input = \
            tf.get_variable(
                name='decoder_input',
                shape=[1, 1, args['embedding_dim']],
                initializer=tf.contrib.layers.xavier_initializer())  # performs xavier init on weights

        start_time = time.time()
        if is_train:
            # Var that contains the model details that are relevant for the training
            self.train_summary = self.build_model(decode_type="stochastic")
            self.train_step = self.build_train_step()
        # Build params for greedy and for beam search
        self.val_summary_greedy = self.build_model(decode_type="greedy")
        self.val_summary_beam = self.build_model(decode_type="beam_search")

        model_time = time.time() - start_time
        self.prt.print_out("It took {}s to build the agent.".format(str(model_time)))
        # The Saver class adds ops to save and restore variables to and from checkpoints.
        # It also provides convenience methods to run these ops.
        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # This is the function that computes R, v, actions, idxs...
    # it's executed in the same way in training or testing
    def build_model(self, decode_type="greedy"):

        args = self.args
        env = self.env
        # Number of problem instances considered at a time
        batch_size = tf.shape(env.input_pnt)[0]
        # input point(s), [?, 11, 2]
        # input_pnt: [batch_size x max_time x 2]
        input_pnt = env.input_pnt
        # Apply embedding to data points
        # encoder_emb_inp: [batch_size, max_time, embedding_dim]
        encoder_emb_inp = self.embedding(input_pnt)
        # Choose the type of decoding that could be greedy, stochastic or BS
        # BS with beam_width = 1 is greedy
        if decode_type == 'greedy' or decode_type == 'stochastic':
            beam_width = 1
        elif decode_type == 'beam_search':
            beam_width = args['beam_width']
        # Reset the environment and passes the beam_width
        env.reset(beam_width)
        # Create a range of batch_size*beam_width integer values and then makes it of shape (?, 1), a column vector
        BatchSequence = \
            tf.expand_dims(
                tf.cast(
                    tf.range(batch_size*beam_width),  # for each prob instance you start with beam_width nodes
                    tf.int64),
                1)

        # Create tensors and lists to fill
        actions_tmp = []
        logprobs = []
        probs = []
        idxs = []

        # Start from depot
        # idx represents an element in the idxs list of tensors that contains sequence solutions as nodes indices
        # idx is a tensor of shape (batch_size * beam_width, 1), full of 10s
        idx = (env.n_nodes-1) * tf.ones([batch_size*beam_width, 1])
        # The last column in input_pnt is for the depot, repeat(tile) the tensor many times
        # action: [batch_size x dim] -> (?, 2)
        action = tf.tile(input_pnt[:, env.n_nodes-1], [beam_width, 1])

        # decoder_state
        # shape (1, 2, ?, 128)
        # stato iniziale del decoder?
        initial_state = tf.zeros([args['rnn_layers'], 2, batch_size*beam_width, args['hidden_dim']])
        # l is a list of tensors, shape (2, ?, 128), unpacks initial_state along the first dimension
        # in this case there is only one because rnn_layers = 1
        l = tf.unstack(initial_state, axis=0)

        # one tuple where each element is a LSTMStateTuple and we have as many elements as layers in the decoder
        # this class is used as initializer
        # Init decoder_state (updated in the loop)
        decoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
            for idx in range(args['rnn_layers'])])

        # start from depot in VRP and from a trainable nodes in TSP
        if args['task_name'] == 'tsp':
            # decoder_input: [batch_size*beam_width x 1 x hidden_dim]
            decoder_input = tf.tile(self.decoder_input, [batch_size * beam_width, 1, 1])
        elif args['task_name'] == 'vrp':
            decoder_input = tf.tile(tf.expand_dims(encoder_emb_inp[:, env.n_nodes-1], 1), [beam_width, 1, 1])
        # the input for the decoder is the embedding of a position of a node? yees
        # decoding loop
        # build context vector used for attention, context has as first dim the repetition of encoder_emb_inp beam_width times
        context = tf.tile(encoder_emb_inp, [beam_width, 1, 1])
        # Perform all the decoding steps specified in decode_len (16 for vrp10),
        # which is the number of time steps of decoding
        for i in range(args['decode_len']):

            logit, prob, logprob, decoder_state = \
                self.decodeStep.step(
                    decoder_input,
                    context,
                    env,
                    decoder_state)
            # idx: [batch_size*beam_width x 1]
            beam_parent = None
            if decode_type == 'greedy':
                # Choose the index corresponding to max prob, along axis 1
                idx = tf.expand_dims(tf.argmax(prob, 1), 1)
            elif decode_type == 'stochastic':
                # Select stochastic actions. idx has shape [batch_size x 1]
                # tf.multinomial sometimes gives numerical errors, so we use our multinomial :(
                def my_multinomial():
                    prob_idx = tf.stop_gradient(prob)
                    prob_idx_cum = tf.cumsum(prob_idx, 1)
                    rand_uni = tf.tile(tf.random_uniform([batch_size,1]),[1,env.n_nodes])
                    # sorted_ind : [[0,1,2,3..],[0,1,2,3..] , ]
                    sorted_ind = tf.cast(tf.tile(tf.expand_dims(tf.range(env.n_nodes), 0), [batch_size, 1]), tf.int64)
                    tmp = tf.multiply(tf.cast(tf.greater(prob_idx_cum, rand_uni), tf.int64), sorted_ind) +\
                        10000*tf.cast(tf.greater_equal(rand_uni, prob_idx_cum), tf.int64)

                    idx = tf.expand_dims(tf.argmin(tmp, 1), 1)
                    return tmp, idx

                # draw samples from a multinomial distribution
                tmp, idx = my_multinomial()
                # check validity of tmp -> True or False -- True mean take a new sample
                tmp_check = \
                    tf.cast(
                        tf.reduce_sum(
                            tf.cast(
                                # check if the sum of tmp values along axis 1 is greater than 10000 * 10 TODO
                                tf.greater(tf.reduce_sum(tmp, 1), (10000*env.n_nodes)-1),
                                tf.int32)
                        ),
                        tf.bool)
                # if tmp_check is true then execute my_multinomial() again and return new samples in tmp, idx
                tmp, idx = tf.cond(tmp_check, my_multinomial, lambda: (tmp, idx))

            # Choose action according to beam search algorithm
            elif decode_type == 'beam_search':
                # If it's the first decoding step
                if i == 0:
                    # BatchBeamSeq: [batch_size*beam_width x 1]
                    # [0,1,2,3,...,127,0,1,...],
                    batchBeamSeq = \
                        tf.expand_dims(
                            tf.tile(
                                tf.cast(tf.range(batch_size), tf.int64),
                                [beam_width]),
                            1)
                    beam_path = []
                    log_beam_probs = []
                    # in the initial decoder step, we want to choose beam_width different branches
                    # log_beam_prob: [batch_size, sourceL]
                    log_beam_prob = \
                        tf.log(
                            tf.split(prob, num_or_size_splits=beam_width, axis=0)[0])

                elif i > 0:
                    log_beam_prob = tf.log(prob) + log_beam_probs[-1]
                    # log_beam_prob:[batch_size, beam_width*sourceL]
                    log_beam_prob = tf.concat(tf.split(log_beam_prob, num_or_size_splits=beam_width, axis=0), 1)

                # topk_prob_val, topk_logprob_ind: [batch_size, beam_width]
                topk_logprob_val, topk_logprob_ind = tf.nn.top_k(log_beam_prob, beam_width)
                # topk_logprob_val , topk_logprob_ind: [batch_size*beam_width x 1]
                topk_logprob_val = \
                    tf.transpose(
                        tf.reshape(
                            tf.transpose(topk_logprob_val),
                            [1, -1]))
                topk_logprob_ind = \
                    tf.transpose(
                        tf.reshape(
                            tf.transpose(topk_logprob_ind),
                            [1, -1]))
                #  idx,beam_parent: [batch_size*beam_width x 1]
                idx = tf.cast(topk_logprob_ind % env.n_nodes, tf.int64) # Which city in route.
                beam_parent = tf.cast(topk_logprob_ind // env.n_nodes, tf.int64) # Which hypothesis it came from.
                # batchedBeamIdx:[batch_size*beam_width]
                batchedBeamIdx = batchBeamSeq + tf.cast(batch_size, tf.int64) * beam_parent
                prob = tf.gather_nd(prob, batchedBeamIdx)

                beam_path.append(beam_parent)
                log_beam_probs.append(topk_logprob_val)
            # Update state
            state = env.step(idx, beam_parent)
            batched_idx = tf.concat([BatchSequence, idx], 1)

            decoder_input = \
                tf.expand_dims(
                    tf.gather_nd(
                        tf.tile(encoder_emb_inp, [beam_width, 1, 1]), batched_idx),
                1)

            logprob = tf.log(tf.gather_nd(prob, batched_idx))
            probs.append(prob)
            idxs.append(idx)  # idx has shape (?, 1) column Tensor
            logprobs.append(logprob)
            # ??? TODO
            action = tf.gather_nd(tf.tile(input_pnt, [beam_width, 1, 1]), batched_idx)
            actions_tmp.append(action)

        if decode_type == 'beam_search':
            # Find paths of the beam search
            tmplst = []
            tmpind = [BatchSequence]
            for k in reversed(range(len(actions_tmp))):

                tmplst = [tf.gather_nd(actions_tmp[k],tmpind[-1])] + tmplst
                tmpind += [tf.gather_nd(
                    (batchBeamSeq + tf.cast(batch_size, tf.int64)*beam_path[k]), tmpind[-1])]
            actions = tmplst
        else:
            actions = actions_tmp

        # Reward_func returns the reward as tour length (negative), having as param 'actions'
        # Which is a tensor that contains the position of the nodes in the solution
        # (positions in the grid are needed to compute euclidean distances)
        R = self.reward_func(actions)

        # critic
        v = tf.constant(0)  # create constant tensor of value 0
        if decode_type == 'stochastic':
            with tf.variable_scope("Critic"):
                with tf.variable_scope("Encoder"):
                    # init states
                    initial_state = tf.zeros([args['rnn_layers'], 2, batch_size, args['hidden_dim']])
                    l = tf.unstack(initial_state, axis=0)
                    rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
                        for idx in range(args['rnn_layers'])])
                    # this is the output?
                    hy = rnn_tuple_state[0][1]

                with tf.variable_scope("Process"):
                    # n_process_blocks is the number of process blocks iteration to run in the critic network
                    for i in range(args['n_process_blocks']):
                        # process is a _call_ object, see AttentionVRPCritic
                        process = self.clAttentionCritic(args['hidden_dim'], _name="P"+str(i))
                        # call clAttentionCritic to output the logit given
                        # given query tensor = hy and ref tensor = encoder_emb_imp
                        e, logit = process(hy, encoder_emb_inp, env)

                        prob = tf.nn.softmax(logit)
                        # hy : [batch_size x 1 x sourceL] * [batch_size  x sourceL x hidden_dim]  ->
                        # [batch_size x h_dim ]
                        hy = tf.squeeze(tf.matmul(tf.expand_dims(prob, 1), e), 1)
                    # end for

                with tf.variable_scope("Linear"):
                    # v is the reward approximation (as in REINFORCE) computed by the critic network
                    # Critic applies ReLu, L1 normalization and then L2
                    v = tf.squeeze(
                            tf.layers.dense(
                                tf.layers.dense(hy, args['hidden_dim'], tf.nn.relu, name='L1'),
                                1, name='L2'),
                        1)
        # idxs are indices of the nodes in the sequence solution
        # actions are represented as destinations (nodes positions x,y)
        return R, v, logprobs, actions, idxs, env.input_pnt, probs

    # Produces a train_step op, that should be run after being returned
    def build_train_step(self):
        '''
        This function returns a train_step op, in which by running it we proceed one training step.
        '''
        args = self.args
        R, v, logprobs, actions, idxs, batch, probs = self.train_summary

        v_nograd = tf.stop_gradient(v)
        R = tf.stop_gradient(R)

        # Compute losses
        actor_loss = \
            tf.reduce_mean(
                tf.multiply((R-v_nograd), tf.add_n(logprobs)),
                0)
        # labels = R, the ground truth output tensor
        # predictions = v, the predicted outputs
        # returns a weighted loss float Tensor
        critic_loss = tf.losses.mean_squared_error(R, v)

        # the args are the learning rates
        actor_optim = tf.train.AdamOptimizer(args['actor_net_lr'])
        critic_optim = tf.train.AdamOptimizer(args['critic_net_lr'])

        # Compute gradients and return list of pairs (gradient, variable)
        # the first param is loss, the value to minimize,
        # the second param is the set of variables that must be updated to minimize this loss
        actor_gra_and_var = actor_optim.compute_gradients(
            actor_loss,
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'))
        critic_gra_and_var = critic_optim.compute_gradients(
            critic_loss,
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'))

        # Clip gradients
        # clip_actor_gra_and_var contains a tensor of couples (gradient, variable)
        clip_actor_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var)
                                  for grad, var in actor_gra_and_var]

        clip_critic_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var)
                                  for grad, var in critic_gra_and_var]

        # Apply gradients to variables
        actor_train_step = actor_optim.apply_gradients(clip_actor_gra_and_var)
        critic_train_step = critic_optim.apply_gradients(clip_critic_gra_and_var)

        train_step = [actor_train_step,
                          critic_train_step,
                          actor_loss,
                          critic_loss,
                          actor_gra_and_var,
                          critic_gra_and_var,
                          R,
                          v,
                          logprobs,
                          probs,
                          actions,
                          idxs]
        return train_step

    # Run a tf session with loaded model
    def Initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.load_model()

    # Load model from latest checkpoint saved
    def load_model(self):
        latest_ckpt = tf.train.latest_checkpoint(self.args['load_path'])
        if latest_ckpt is not None:
            self.saver.restore(self.sess, latest_ckpt)

    # Function used to evaluate model on inference for one sample
    def evaluate_single(self, eval_type='greedy'):
        start_time = time.time()
        avg_reward = []

        if eval_type == 'greedy':
            summary = self.val_summary_greedy
        elif eval_type == 'beam_search':
            summary = self.val_summary_beam
        self.dataGen.reset()
        for step in range(self.dataGen.n_problems):
            # get the next batch of problems for testing
            data = self.dataGen.get_test_next()
            # summary contains details of the built model, feed_dict is used for the input values
            # since it can override values in the TensorFlow graph
            # it returns a tensor with same shape as 'summary' with updated values in the leaves
            R, v, logprobs, actions, idxs, batch, _ = \
                self.sess.run(
                    summary,
                    feed_dict={self.env.input_data: data, self.decodeStep.dropout: 0.0})

            if eval_type == 'greedy':
                avg_reward.append(R)
                R_ind0 = 0

            elif eval_type == 'beam_search':
                # R : [batch_size x beam_width]
                R = np.concatenate(np.split(np.expand_dims(R, 1), self.args['beam_width'], axis=0), 1)
                # returns the min in the array (or along one dimension)
                R_val = np.amin(R, 1, keepdims=False)
                # takes the index corresponding to minimum
                R_ind0 = np.argmin(R, 1)[0]
                avg_reward.append(R_val)

            # sample decode
            if step % int(self.args['log_interval']) == 0:
                example_output = []
                example_input = []
                for i in range(self.env.n_nodes):
                    example_input.append(list(batch[0, i, :]))
                for idx, action in enumerate(actions):
                    example_output.append(list(action[R_ind0 * np.shape(batch)[0]]))
                self.prt.print_out('\n\nVal-Step of {}: {}'.format(eval_type,step))
                self.prt.print_out('\nExample test input: {}'.format(example_input))
                self.prt.print_out('\nExample test output: {}'.format(example_output))
                self.prt.print_out('\nExample test reward: {} - best: {}'.format(R[0],R_ind0))

        end_time = time.time() - start_time

        # Finished going through the iterator dataset.
        self.prt.print_out('\nValidation overall avg_reward: {}'.format(np.mean(avg_reward)) )
        self.prt.print_out('Validation overall reward std: {}'.format(np.sqrt(np.var(avg_reward))) )

        self.prt.print_out("Finished evaluation with %d steps in %s." % (step\
                           ,time.strftime("%H:%M:%S", time.gmtime(end_time))))

    def evaluate_batch(self, eval_type='greedy'):
        self.env.reset()
        if eval_type == 'greedy':
            summary = self.val_summary_greedy
            beam_width = 1
        elif eval_type == 'beam_search':
            summary = self.val_summary_beam
            beam_width = self.args['beam_width']

        data = self.dataGen.get_test_all()
        start_time = time.time()
        # R contains the reward obtained for each test problem (1000 rewards)
        # batch has shape (1000, 11, 2)
        R, v, logprobs, actions, idxs, batch, _ = \
            self.sess.run(
                summary,
                feed_dict={self.env.input_data: data,
                            self.decodeStep.dropout: 0.0})
        # Split the column vector R
        R = np.concatenate(np.split(np.expand_dims(R, 1), beam_width, axis=0), 1)
        # Take min reward in array along axis 1, quindi per riga, R has now shape (1000, ) -> row vector
        R = np.amin(R, 1, keepdims=False)

        end_time = time.time() - start_time
        # Print evaluation results, average reward and std deviation from reward
        self.prt.print_out('Average of {} in batch-mode: {} -- std {} -- time {} s'.format(
            eval_type, np.mean(R), np.sqrt(np.var(R)), end_time))

    # Used for inference on an already trained model, evaluation is performed every fixed interval
    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
            self.evaluate_batch('beam_search')
        elif infer_type == 'single':
            self.evaluate_single('greedy')
            self.evaluate_single('beam_search')
        self.prt.print_out("##################################################################")

    def run_train_step(self):
        # Get next train iteration and run it
        data = self.dataGen.get_train_next()
        train_results = \
            self.sess.run(
                self.train_step,  # from build_train_step()
                 feed_dict={
                     self.env.input_data: data,
                     self.decodeStep.dropout: self.args['dropout']
                 })
        return train_results  # summary var in main
