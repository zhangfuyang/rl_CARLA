import tensorflow as tf
from src.ops import fully_connected, batch_norm
from src.network.network import BaseNetwork
from src.utils import discretize


class ActorNetwork(BaseNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, action_type, scope='Actor'):
        super(ActorNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)
        self.action_bound = action_bound
        self.action_type = action_type
        self.scope = scope

        # Actor network
        self.inputs, self.phase, self.outputs, self.scaled_outputs = self.build_network('eval')
        self.net_params = tf.trainable_variables(scope=self.scope)

        # Target network
        self.target_inputs, self.target_phase, self.target_outputs, self.target_scaled_outputs = self.build_network('target')
        self.target_net_params = tf.trainable_variables(scope=self.scope)[len(self.net_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        self.update_target_bn_params = \
            [self.target_net_params[i].assign(self.net_params[i]) for i in range(len(self.target_net_params)) if self.target_net_params[i].name.startswith('BatchNorm')]


        # Combine dnetScaledOut/dnetParams with criticToActionGradient to get actorGradient
        # Temporary placeholder action gradient
        if self.action_type == 'Continuous':
            self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])
        else:
            self.action_gradients = tf.placeholder(tf.float32, [None, 1])

        self.actor_gradients = tf.gradients(self.outputs, self.net_params, -self.action_gradients)
        self.check_gradient = tf.gradients(self.outputs, self.net_params[-2])

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.net_params))

        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

    def build_network(self, scope):
        if self.action_type == 'Continuous':
            inputs = tf.placeholder(tf.float32, shape=(None,) + self.state_dim)
            phase = tf.placeholder(tf.bool)
            with tf.variable_scope(self.scope):
                with tf.variable_scope(scope):
                    net = fully_connected(inputs, 256, activation_fn=tf.nn.relu)
                    net = fully_connected(net, 256, activation_fn=tf.nn.relu)
                    net = fully_connected(net, 128, activation_fn=tf.nn.relu)
                    net = fully_connected(net, 64, activation_fn=tf.nn.relu)
                    # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
                    outputs = fully_connected(net, self.action_dim, activation_fn=tf.tanh)#, weights_initializer=tf.random_uniform_initializer(-3e-5, 3e-5))
                    scaled_outputs = tf.multiply(outputs, self.action_bound) # Scale output to [-action_bound, action_bound]
        else:
            inputs = tf.placeholder(tf.float32, shape=(None,) + self.state_dim)
            phase = tf.placeholder(tf.bool)
            with tf.variable_scope(self.scope):
                with tf.variable_scope(scope):
                    net = fully_connected(inputs, 400, activation_fn=tf.nn.relu)
                    net = fully_connected(net, 256, activation_fn=tf.nn.relu)
                    net = fully_connected(net, 128, activation_fn=tf.nn.relu)
                    net = fully_connected(net, 32, activation_fn=tf.nn.relu)
                    # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
                    outputs = fully_connected(net, 1)#, weights_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))
                    scaled_outputs = discretize(outputs, self.action_dim)
            
        return inputs, phase, outputs, scaled_outputs

    def train(self, *args):
        # args [inputs, action_gradients, phase]
        return self.sess.run(self.optimize, feed_dict={
            self.inputs: args[0],
            self.action_gradients: args[1],
            self.phase: True
        })

    def check_(self, *args):
        grad_, outputs_ = self.sess.run([self.check_gradient, self.outputs], feed_dict={
            self.inputs: args[0],
            self.phase: False
        })
        return [grad_[0], outputs_[0]]

    def predict(self, *args):
        return self.sess.run(self.scaled_outputs, feed_dict={
            self.inputs: args[0],
            self.phase: False
        })

    def predict_target(self, *args):
        return self.sess.run(self.target_scaled_outputs, feed_dict={
            self.target_inputs: args[0],
            self.target_phase: False,
        })

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(BaseNetwork):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, num_actor_vars, action_type,
                 scope='Critic'):
        super(CriticNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)
        self.action_bound = action_bound
        self.action_type = action_type
        self.scope = scope

        # Critic network
        self.inputs, self.phase, self.action, self.outputs = self.build_network('eval')
        self.net_params = tf.trainable_variables(scope=self.scope)

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_outputs = self.build_network('target')
        self.target_net_params = tf.trainable_variables(self.scope)[len(self.net_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        self.update_target_bn_params = \
            [self.target_net_params[i].assign(self.net_params[i]) for i in range(len(self.target_net_params)) if self.target_net_params[i].name.startswith('BatchNorm')]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.outputs))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.net_params)

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.outputs, self.action)

    def build_network(self, scope):
        inputs = tf.placeholder(tf.float32, shape=(None,) + self.state_dim)
        phase = tf.placeholder(tf.bool)
        if self.action_type == 'Continuous':
            action = tf.placeholder(tf.float32, [None, self.action_dim])
            with tf.variable_scope(self.scope):
                with tf.variable_scope(scope):
                    net = fully_connected(inputs, 400, activation_fn=tf.nn.relu)
                    net = fully_connected(tf.concat([net, action], 1), 300, activation_fn=tf.nn.relu)
                    net = fully_connected(net, 128, activation_fn=tf.nn.relu)
                    outputs = fully_connected(net, 1)#, weights_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))
        else:
            action = tf.placeholder(tf.float32, [None, 1])
            with tf.variable_scope(self.scope):
                with tf.variable_scope(scope):
                    net = fully_connected(inputs, 400, activation_fn=tf.nn.relu)
                    net = fully_connected(tf.concat([net, action], 1), 300, activation_fn=tf.nn.relu)
                    outputs = fully_connected(net, 1)#, weights_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))

        return inputs, phase, action, outputs

    def train(self, *args):
        # args (inputs, action, predicted_q_value, phase)
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
            self.phase: True
        })

    def predict(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.phase: False
        })

    def predict_target(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: args[0],
            self.target_action: args[1],
            self.target_phase: False
        })

    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: False
        })

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)
