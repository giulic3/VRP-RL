import argparse
import os
import numpy as np
from tqdm import tqdm # just a progress bar
import tensorflow as tf
import time
from configs import ParseParams
from shared.decode_step import RNNDecodeStep
from model.attention_agent import RLAgent


# load different components according to the type of the task: vrp or tsp
def load_task_specific_components(task):
    '''
    This function load task-specific libraries
    '''
    if task == 'tsp':
        from TSP.tsp_utils import DataGenerator, Env, reward_func
        from shared.attention import Attention

        AttentionActor = Attention
        AttentionCritic = Attention

    elif task == 'vrp':
        from VRP.vrp_utils import DataGenerator, Env, reward_func
        from VRP.vrp_attention import AttentionVRPActor, AttentionVRPCritic

        AttentionActor = AttentionVRPActor
        AttentionCritic = AttentionVRPCritic

    else:
        raise Exception('Task is not implemented')

    return DataGenerator, Env, reward_func, AttentionActor, AttentionCritic


def main(args, prt):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Load task specific classes
    DataGenerator, Env, reward_func, AttentionActor, AttentionCritic = \
        load_task_specific_components(args['task_name'])
    # Init the object that generates problems for training and test (in vrp_utils.py)
    dataGen = DataGenerator(args)
    dataGen.reset()
    env = Env(args)
    # Create a RL agent -> main f to build the model
    # It takes approximately 4 minutes
    agent = RLAgent(args,
                    prt,  # print controller used to write logs
                    env,  # Environment
                    dataGen,
                    reward_func,  # reward function
                    AttentionActor,
                    AttentionCritic,
                    is_train=args['is_train'])  # Can be used for training or inference
    # Run a tf session
    agent.Initialize(sess)

    # Train or evaluate
    start_time = time.time()
    if args['is_train']:
        prt.print_out('Training started ...')
        # Takes the time at the beginning of an iteration
        train_time_beg = time.time()
        # Start training
        for step in range(args['n_train']):
            # 'summary' is a list
            summary = agent.run_train_step()
            # Training variables (printed at each it)
            _, _, actor_loss_val, critic_loss_val, actor_gra_and_var_val, critic_gra_and_var_val,\
                R_val, v_val, logprobs_val, probs_val, actions_val, idxs_val = summary
            # save checkpoint at given intervals of time
            if step % args['save_interval'] == 0:
                agent.saver.save(sess, args['model_dir']+'/model.ckpt', global_step=step)
            # log to console at given intervals of time
            if step % args['log_interval'] == 0:
                train_time_end = time.time() - train_time_beg
                prt.print_out('Train Step: {} -- Time: {} -- Train reward: {} -- Value: {}'
                      .format(step, time.strftime("%H:%M:%S", time.gmtime(\
                        train_time_end)), np.mean(R_val), np.mean(v_val)))
                prt.print_out('    actor loss: {} -- critic loss: {}'
                      .format(np.mean(actor_loss_val), np.mean(critic_loss_val)))
                train_time_beg = time.time()
            if step % args['test_interval'] == 0:
                agent.inference(args['infer_type'])

    else:  # Inference (find solution for test problems, don't perform training again)
        prt.print_out('Evaluation started ...')
        agent.inference(args['infer_type'])  # batch by default

    prt.print_out('Total time is {}'.format(\
        time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))


if __name__ == "__main__":
    args, prt = ParseParams()
    # Random
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        prt.print_out("# Set random seed to %d" % random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    # Clears the default TensorFlow graph stack and resets the global default graph
    # the default graph is a property of the current thread
    tf.reset_default_graph()

    main(args, prt)
