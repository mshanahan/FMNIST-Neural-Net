#author: Michael Shanahan 42839964
import tensorflow as tf
import numpy as np

#split the data and labels
def split_data(data, labels, proportion):
  #code from Canvas: written by Paul Quint, refactored by Brandon Geren
    size = data.shape[0]
    np.random.seed(42)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], data[s[split_idx:]], labels[s[:split_idx]], labels[s[split_idx:]]
def white_hot(labels):
  shape = label_shape = ( len(labels), 10 ) #n rows, 10 columns
  new_array = np.zeros(label_shape)
  new_array[np.arrange(len(labels)),labels] = 1
  return new_array