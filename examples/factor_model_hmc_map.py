#!/usr/bin/env python
"""Logistic factor analysis on MNIST.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import os
import tensorflow as tf

from edward.models import Empirical, Bernoulli, Normal
from progressbar import ETA, Bar, Percentage, ProgressBar
from scipy.misc import imsave
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  z = tf.expand_dims(z, 0)
  net = slim.fully_connected(z, 28 * 28, activation_fn=None)
  net = slim.flatten(net)
  return net


ed.set_seed(42)

N = 1
d = 10  # latent variable dimension
DATA_DIR = "data/mnist"
IMG_DIR = "img"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)

# DATA
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
x_train, _ = mnist.train.next_batch(N)

# MODEL
with tf.variable_scope("model"):
  z = Normal(mu=tf.zeros([N, d]), sigma=tf.ones([N, d]))
  logits = generative_network(z.value())
  x = Bernoulli(logits=logits)

# INFERENCE
T = int(100 * 1000)
qz = Empirical(params=tf.Variable(tf.random_normal([T, d])))

# TODO
inference_e = ed.HMC({z: qz}, data={x: x_train})
inference_e.initialize()

# inference_m = ed.MAP({z: qz}, data={x: x_train})
# inference_m = ed.MAP({}, data={x: x_train}, cond_vars={z: qz})
inference_m = ed.MAP({},
    data={x: x_train, z: tf.gather(qz.params, inference_e.t)})
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference_m.initialize(optimizer=optimizer, scope="model")

# CRITICISM. Build posterior predictive check.
with tf.variable_scope("model", reuse=True):
  # p_rep = tf.sigmoid(logits)
  # Use last sampled output from HMC.
  z_sample = tf.gather(qz.params, tf.maximum(inference_e.t - 1, 0))
  p_rep = tf.expand_dims(tf.sigmoid(generative_network(z_sample)), 0)

# Run inference.
init = tf.initialize_all_variables()
init.run()

n_iter_per_epoch = 100
n_epoch = T // n_iter_per_epoch
for epoch in range(n_epoch):
  avg_loss = 0.0

  widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
  pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
  pbar.start()
  for t in range(n_iter_per_epoch):
    pbar.update(t)
    info_dict_e = inference_e.update()
    info_dict_m = inference_m.update()
    avg_loss += info_dict_m['loss']

  print("Acceptance Rate:")
  print(info_dict_e['accept_rate'])

  # Take average over all ELBOs during the epoch, and over minibatch
  # of data points (images).
  avg_loss = avg_loss / n_iter_per_epoch
  avg_loss = avg_loss / N

  # Print a lower bound to the average marginal likelihood for an
  # image.
  print("log p(x) >= {:0.3f}".format(avg_loss))

  # Visualize hidden representations.
  imgs = p_rep.eval()
  for b in range(N):
    imsave(os.path.join(IMG_DIR, '%d.png') % b,
           imgs[b].reshape(28, 28))
