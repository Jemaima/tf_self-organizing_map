import numpy as np
import tensorflow as tf


class SOMNetwork():
    def __init__(self, inp_dim, dim=3, sigma=None, l_r=0.5, min_lr=0.001, tau2=100000, dtype=tf.float32):
        if not sigma:
            sigma = dim / 2
        self.dtype = dtype
        self.dim = tf.constant(dim, dtype=tf.int64)
        self.l_r = tf.constant(l_r, dtype=dtype, name='l_r')
        self.min_lr = tf.constant(min_lr, dtype=self.dtype, name='min_learning_rate')
        self.sigma = tf.constant(sigma, dtype=dtype, name='sigma')
        self.min_sigma = tf.constant(inp_dim / 100, dtype=dtype, name='min_sigma')
        # self.min_sigma = tf.constant(sigma * np.exp(-np.log(sigma)), dtype=dtype, name='min_sigma')
        self.tau1 = tf.constant(100000 / np.log(sigma), dtype=dtype, name='tau1')
        self.tau2 = tf.constant(tau2, dtype=dtype, name='tau2')

        self.x = tf.placeholder(shape=[inp_dim], dtype=dtype, name='inp_x')
        self.n_iter = tf.placeholder(dtype=dtype, name='iteration')
        self.w = tf.Variable(tf.random_uniform([dim * dim, inp_dim], minval=-1, maxval=1), dtype=dtype, name='weights')
        self.pos = tf.where(tf.fill([dim, dim], True))

    def _competition(self):
        with tf.name_scope('competition') as scope:
            dist = tf.sqrt(tf.reduce_sum(tf.square(self.x - self.w), axis=1))
        return tf.argmin(dist, axis=0)

    def train_op(self):
        win_ind = self._competition()
        with tf.name_scope('cooperation') as scope:
            coop_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(self.pos -
                                                                [win_ind // self.dim,
                                                                 win_ind - win_ind // self.dim * self.dim],
                                                                dtype=self.dtype)), axis=1))
            sigma = tf.cond(self.sigma <= self.min_sigma, lambda: self.min_sigma,
                            lambda: self.sigma * tf.exp(-self.n_iter / self.tau1))
            # active for changing map
            tnh = tf.exp(-tf.square(coop_dist) / (2 * tf.square(sigma)))
        with tf.name_scope('adaptation') as scope:
            l_r = self.l_r * tf.exp(-self.n_iter / self.tau2)
            l_r = tf.cond(l_r <= self.min_lr, lambda: self.min_lr, lambda: l_r)
            delta = tf.transpose(l_r * tnh * tf.transpose(self.x - self.w))
            train_op = tf.assign(self.w, self.w + delta)
        return train_op, tnh
