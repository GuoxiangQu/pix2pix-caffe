"""
Batch Loader by Donny You
"""

import cv2
import numpy as np
import numpy.random as nr
from random import shuffle
import os

class BatchLoader(object):

    def __init__(self, file_path, batch_size):
        self.batch_size = batch_size
        self.im_list = self.image_dir_processor(file_path)
        self.idx = 0
        self.data_num = len(self.im_list)
        self.rnd_list = np.arange(self.data_num)
        shuffle(self.rnd_list)

    def next_batch(self):
        batch_images_A = []
        batch_images_B = []

        for i in xrange (self.batch_size):
            if self.idx != self.data_num:
                cur_idx = self.rnd_list[self.idx]
                im_path = self.im_list[cur_idx]
                image_a = cv2.imread("./a/" + im_path)
                image_b = cv2.imread("./b/" + im_path)
                image_a = (image_a - 127.5) / 128
                image_b = (image_b - 127.5) / 128
                batch_images_A.append(image_a)
                batch_images_B.append(image_b)

                self.idx +=1
            else:
                self.idx = 0
                shuffle(self.rnd_list)
                cur_idx = self.rnd_list[self.idx]
                im_path = self.im_list[cur_idx]
                image_a = cv2.imread("./a/" + im_path)
                image_b = cv2.imread("./b/" + im_path)
                image_a = (image_a - 127.5) / 128
                image_b = (image_b - 127.5) / 128
                batch_images_A.append(image_a)
                batch_images_B.append(image_b)
                self.idx += 1


        batch_images_A = np.array(batch_images_A).astype(np.float32)
        batch_labels_B = np.array(batch_images_B).astype(np.float32)
        return batch_images_A, batch_images_B 
        
    def image_dir_processor(self, file_path):
        im_path_list = []
        if not os.path.exists(file_path):
            print "File %s not exists." % file_path
            exit()

        with open(file_path, "r") as fr:
            for line in fr.readlines():
                terms = line.rstrip()
                im_path_list.append(terms)

        return im_path_list
