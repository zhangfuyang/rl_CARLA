from __future__ import print_function

import os

import scipy

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

from environment.carla.agent.agent import Agent
from environment.imitation.imitation_learning_network import load_imitation_learning_network


class ImitationLearning(Agent):

    def __init__(self, sess, memory_fraction=0.25, image_cut=(115, 510)):

        Agent.__init__(self)

        self.dropout_vec = [1.0] * 8 + [0.7] * 2

        self._image_size = (88, 200, 3)
        self._sess = sess

        self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                            self._image_size[1],
                                                            self._image_size[2]],
                                            name="input_image")

        self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):
            self._network_tensor = load_imitation_learning_network(self._input_images,
                                                                   self._image_size, self._dout)

        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'

        self._image_cut = image_cut
        self.variables_to_restore = tf.global_variables()

    def load_model(self):

        saver = tf.train.Saver(self.variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0

        return ckpt

    def compute_feature(self, sensor_data):

        rgb_image = sensor_data['CameraRGB'].data
        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)
        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))

        feedDict = {self._input_images: image_input, self._dout: [1] * len(self.dropout_vec)}

        output_vector = self._sess.run(self._network_tensor, feed_dict=feedDict)

        return output_vector[0]
