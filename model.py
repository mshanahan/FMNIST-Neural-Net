#author: Michael Shanahan 42839964
import util
import tensorflow as tf
import numpy as np

def my_model(layer_counts):
  #layer_counts: a 4d array of integers
  #[0]: hidden 1 size, [1]: hidden 2 size
  with tf.name_scope('my_model') as scope:
    hidden_1 = tf.layers.Dense(
      layer_counts[0],
      activation = tf.nn.relu,
      bias_regularizer = tf.contrib.layers.l2_regularizer(1),
      kernel_regularizer = tf.contrib.layers.l2_regularizer(1),
      name = 'hidden_1')
      
    hidden_2 = tf.layers.Dense(
      layer_counts[1],
      activation = tf.nn.relu,
      bias_regularizer = tf.contrib.layers.l2_regularizer(1),
      kernel_regularizer = tf.contrib.layers.l2_regularizer(1),
      name = 'hidden_2')
      
    output = tf.layers.Dense(
      10,
      activation = tf.nn.relu,
      bias_regularizer = tf.contrib.layers.l2_regularizer(1),
      kernel_regularizer = tf.contrib.layers.l2_regularizer(1),
      name = 'output')
      
    return hidden_1,hidden_2,output