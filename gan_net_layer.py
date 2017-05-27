"""
For image data augment layer

Revised from CaoChunShui Gait_data_layer.

by Donny You

2016/12/08
"""

import sys
sys.path.append("/home/donny/test_data_layer/caffe/python")
import caffe
import numpy as np
import numpy.random as nr
import os
import yaml

from batch_loader import BatchLoader


class GanNetTrain(caffe.Layer):
    """Data layer for training"""   
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.batch_size = layer_params['batch_size']
        self.image_file = layer_params['image_file']
        self.batch_loader = BatchLoader(self.image_file, self.batch_size)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.images_a
        top[1].data[...] = self.images_b
        top[2].data[...] = self.label_true
        top[3].data[...] = self.label_false

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        images_A, images_B = self.batch_loader.next_batch()
        images_A = np.array(images_A)
        images_B = np.array(images_B)
        images_A = images_A.transpose((0, 3, 1, 2))
        images_B = images_B.transpose((0, 3, 1, 2))

        self.images_a = images_A
        self.images_b = images_B
        self.label_true = np.ones((self.batch_size, 1), dtype='float32')
        self.label_false = np.zeros((self.batch_size, 1), dtype='float32')

        top[0].reshape(*self.images_a.shape)
        top[1].reshape(*self.images_b.shape)
        top[2].reshape(*self.label_true.shape)
        top[3].reshape(*self.label_false.shape)
