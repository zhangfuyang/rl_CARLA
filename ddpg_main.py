"""
Implementation of DDPG - Deep Deterministic Policy Gradient Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 and MountainCarContinuous-v0 OpenAI gym task
"""

import numpy as np
import datetime
import tensorflow as tf

from tqdm import tqdm

from environment.env import Env
from environment.carla.client import make_carla_client
from environment.carla.tcp import TCPConnectionError
from src.agent.ddpg_agent import DDPGAgent
from src.network.ddpg_network import CriticNetwork, ActorNetwork
from src.replaybuffer import ReplayBuffer
from src.explorationnoise import OrnsteinUhlenbeckProcess, GreedyPolicy
import time

flags = tf.app.flags

# ================================
#    UTILITY PARAMETERS
# ================================
# environment name
flags.DEFINE_string('env_name', 'carla_soft', 'environment name.')
flags.DEFINE_boolean('env_render', True, 'whether render environment (display).')
flags.DEFINE_integer('port', 2000, 'simulation listening port')
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
RANDOM_SEED = 1234
FPS = 10

# ================================
#    TRAINING PARAMETERS
# ================================
flags.DEFINE_integer('mini_batch', 256, 'mini batch size for training.')

# render interval
RENDER_INTERVAL = 100
# Learning rates actor and critic
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
# Maximum number of episodes
MAX_EPISODES = 200000
# Maximum number of steps per episode
MAX_STEPS_EPISODE = 5000
# warmup steps.
WARMUP_STEPS = 3000
# Exploration duration
EXPLORATION_EPISODES = 20000
# Discount factor
GAMMA = 0.99
# Soft target update parameter
TAU = 0.001
# Size of replay buffer
BUFFER_SIZE = 1000000
# Exploration noise variables Ornstein-Uhlenbeck variables
OU_THETA = 10
OU_MU = 0.
OU_SIGMA = 0.4
# Explorationnoise for greedy policy
MIN_EPSILON = 0.1
MAX_EPSILON = 1

#================
# parameters for evaluate.
#================
# evaluate periods
EVAL_PERIODS = 100
# evaluate episodes
EVAL_EPISODES = 10

# store model periods
MODEL_STORE_PERIODS = 30

FLAGS = flags.FLAGS

# whether to print on the screen
DETAIL = True

# Directory for storing gym results
MONITOR_DIR = './results/{}/{}/pic_ddpg'.format(FLAGS.env_name, DATETIME)
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/{}/{}/tf_ddpg'.format(FLAGS.env_name, DATETIME)
# Directory for storing model
MODEL_DIR = './results/{}/{}/tf_net'.format(FLAGS.env_name, DATETIME)


# ================================
#    MAIN
# ================================
def main(_):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        env = Env(MONITOR_DIR, RANDOM_SEED, FPS, sess)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = env.observation_space.shape
        try: 
            action_dim = env.action_space.shape[0]
            action_bound = env.action_space.high
            # Ensure action bound is symmetric
            assert(np.all(env.action_space.high == -env.action_space.low))
            action_type = 'Continuous'
        except:
            action_dim = env.action_space.n
            action_bound = None
            action_type = 'Discrete'

        print(action_type)
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU, action_type)

        critic = CriticNetwork(sess, state_dim, action_dim, action_bound,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), action_type)

        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        if action_type == 'Continuous':
            noise = OrnsteinUhlenbeckProcess(OU_THETA, mu=OU_MU, sigma=OU_SIGMA, n_steps_annealing=EXPLORATION_EPISODES,
                                             size=action_dim)
        else:
            noise = GreedyPolicy(action_dim, EXPLORATION_EPISODES, MIN_EPSILON, MAX_EPSILON)

        agent = DDPGAgent(sess, action_type, actor, critic, GAMMA, env, replay_buffer, noise=noise,
                          exploration_episodes=EXPLORATION_EPISODES, max_episodes=MAX_EPISODES,
                          max_steps_episode=MAX_STEPS_EPISODE, warmup_steps=WARMUP_STEPS,
                          mini_batch=FLAGS.mini_batch, eval_episodes=EVAL_EPISODES, eval_periods=EVAL_PERIODS,
                          env_render=FLAGS.env_render, summary_dir=SUMMARY_DIR, model_dir=MODEL_DIR,
                          model_store_periods=MODEL_STORE_PERIODS, detail=DETAIL, render_interval=RENDER_INTERVAL)

        while True:
            try:
                with make_carla_client('localhost', FLAGS.port) as client:
                    env.connected(client)
                    agent.train()
            except TCPConnectionError as error:
                print(error)
                time.sleep(5.0)




if __name__ == '__main__':
    tf.app.run()


