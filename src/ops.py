# cnn and fully_connected layers.
import tensorflow as tf

def fully_connected(inputs, output_size, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),\
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.001), biases_initializer=tf.constant_initializer(0.0)):
    return tf.contrib.layers.fully_connected(inputs, output_size, activation_fn=activation_fn, \
            weights_initializer=weights_initializer, weights_regularizer=weights_regularizer, biases_initializer=biases_initializer)

def batch_norm(inputs, phase):
    return tf.contrib.layers.batch_norm(inputs, center=True, scale=True, is_training=phase)
