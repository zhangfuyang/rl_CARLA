import tensorflow as tf

class BaseAgent(object):
    def __init__(self, sess, env, replay_buffer, noise=None, exploration_episodes=10000, max_episodes=10000, max_steps_episode=10000, warmup_steps=5000,\
            mini_batch=32, eval_episodes=10, eval_periods=100, env_render=False, summary_dir=None):
        """
        Base agent, provide basic functions. 
        Args:
            sess: tf.Session(). 
            env: openai gym environment. could be a wrapper.
            replay_buffer: replay_buffer for sampling. 
            noise: noise added to action for exploration. 
            exploration_episodes: maximum episodes for training with noise.
            max_episodes: maximum episodes for training.
            max_steps_episode: maximum steps per episode.
            mini_batch: mini batch size in the training.
            eval_episodes: number of episodes to evaluate current model.
            eval_periods: periods to evaluate model.
            env_render: whether display observation.
            summary_dir: folder to store summaries of algorithm.
        """
        self.sess = sess
        self.env = env
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.exploration_episodes = exploration_episodes
        self.max_episodes = max_episodes
        self.max_steps_episode = max_steps_episode
        self.warmup_steps = warmup_steps
        self.mini_batch = mini_batch
        self.eval_episodes = eval_episodes
        self.eval_periods = eval_periods
        self.env_render = env_render
        self.summary_dir = summary_dir
        self.saver = tf.train.Saver()

        # Initialize Tensorflow variables
        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

    def train(self):
        """
        Train the model. 
        """
        raise NotImplementedError("train() method should be implemented")

    def evaluate(self, cur_episode):
        """
        evaluate the model.
        """
        raise NotImplementedError("evaluate() method should be implemented")



