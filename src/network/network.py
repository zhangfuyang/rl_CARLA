
class BaseNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        """
        base network for actor and critic network.
        Args:
            sess: tf.Session()
            state_dim: env.observation_space.shape
            action_dim: env.action_space.shape[0]
            learning_rate: learning rate for training
            tau: update parameter for target.
        """
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

    def build_network(self, scope):
        """
        build network.
        """
        raise NotImplementedError("build newtork first!")

    def train(self, *args):
        raise NotImplementedError("train network!")

    def predict(self, *args):
        raise NotImplementedError("predict output for network!")

    def predict_target(self, *args):
        raise NotImplementedError("predict output for target network!")

    def update_target_network(self):
        raise NotImplementedError("update target network!")

    def get_num_trainable_vars(self):
        raise NotImplementedError("update target network!")




