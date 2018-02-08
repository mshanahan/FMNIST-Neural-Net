#author: Michael Shanahan 42839964
import util
import tensorflow as tf
import numpy as np

def my_model(layer_counts, inputs):
  #layer_counts: a 4d array of integers
  #[0]: hidden 1 size, [1]: hidden 2 size
  with tf.name_scope('my_model') as scope:
    divided_inputs = inputs / 255
  
    hidden_1 = tf.layers.dense(
      divided_inputs,
      layer_counts[0],
      activation = tf.nn.relu,
      bias_regularizer = tf.contrib.layers.l2_regularizer(1.0),
      kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
      name = 'hidden_1')
      
    hidden_2 = tf.layers.dense(
      hidden_1,
      layer_counts[1],
      activation = tf.nn.relu,
      bias_regularizer = tf.contrib.layers.l2_regularizer(1.0),
      kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
      name = 'hidden_2')

    hidden_3 = tf.layers.dense(
      hidden_2,
      layer_counts[2],
      activation = tf.nn.relu,
      bias_regularizer = tf.contrib.layers.l2_regularizer(1.0),
      kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
      name = 'hidden_3')
      
    output = tf.layers.dense(
      hidden_3,
      10,
      activation = tf.nn.relu,
      name = 'output')
      
    return output