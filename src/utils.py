import numpy as np
import tensorflow as tf

def discretize(value, num_actions):
    discretization = tf.round(value)
    discretization = tf.minimum(tf.constant(num_actions-1, dtype=tf.float32), tf.maximum(tf.constant(0, dtype=tf.float32), tf.to_float(discretization)))
    return tf.to_int32(discretization)
