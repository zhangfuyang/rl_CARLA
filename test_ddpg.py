import numpy as np
import tensorflow as tf

from environment.env import Env
from environment.carla.client import make_carla_client
from environment.carla.tcp import TCPConnectionError
from src.agent.ddpg_agent import DDPGAgent
from src.network.ddpg_network import CriticNetwork, ActorNetwork
import time

ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
TAU = 0.001
GAMMA = 0.99
RANDOM_SEED = 123
flags = tf.app.flags
flags.DEFINE_string('model_path', 'ckpt', '')
FLAGS = flags.FLAGS


def main(_):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        env = Env(MONITOR_DIR=None, SEED=RANDOM_SEED, FPS=10, sess=sess)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        assert(np.all(env.action_space.high == -env.action_space.low))
        action_type = 'Continuous'

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU, action_type)

        critic = CriticNetwork(sess, state_dim, action_dim, action_bound,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(),
                               action_type)

        agent = DDPGAgent(sess, action_type, actor, critic, GAMMA, env, Inference=True, Inference_net_dir=FLAGS.model_path)

        while True:
            try:
                with make_carla_client('localhost', 2000) as client:
                    env.connected(client)
                    for i in range(30):
                        state = env.reset()
                        terminal = False
                        while not terminal:
                            action = actor.predict(np.expand_dims(state, 0))[0]
                            if action[1] >= 0:
                                step_ = {'steer': action[0], 'acc': action[1], 'brake': 0.0}
                            else:
                                step_ = {'steer': action[0], 'acc': 0.0, 'brake': action[1]}
                            state, reward, terminal, info = env.step(step_)

            except TCPConnectionError as error:
                print(error)
                time.sleep(1)

if __name__ == '__main__':
    tf.app.run()
