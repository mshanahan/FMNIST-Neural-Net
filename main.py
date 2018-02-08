#author: Michael Shanahan 42839964
import model
import util
import tensorflow as tf
import numpy as np

#code to set flags by Paul Quint
flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', 'homework_1', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def main(argv):
  #load data
  fmnist_data = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
  fmnist_labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')

  #split data
  VALID_PROPORTION = 0.1 #proportion of training data used for validation
  valid_data, train_data, valid_labels, train_labels = util.split_data(fmnist_data, fmnist_labels, VALID_PROPORTION)
  
  #count data
  valid_count = valid_data.shape[0]
  train_count = train_data.shape[0]

  #specify model
  input_placeholder = tf.placeholder(tf.float32, [None,784], name='input_placeholder')
  my_network = model.my_model([64,64,64], input_placeholder)

  #define classification loss
  #code adapted from Paul Quint's hackathon 3
  labels = tf.placeholder(tf.float32, [None, 10], name='labels')
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=my_network)
  confusion_matrix_op = tf.confusion_matrix(tf.argmax(labels, axis=1), tf.argmax(my_network, axis=1), num_classes=10)
  REG_COEFF = 0.0001
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)

  #set up training and saving
  #code adapted from Paul Quint's hackathon 3
  global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
  saver = tf.train.Saver()
  sum_cross_entropy = tf.reduce_mean(cross_entropy)
  
  best_epoch = 0
  best_valid_ce = 0
  best_train_ce = 0
  best_classification_rate = 0
  epochs_since_best = 0
  
  #run the actual training
  #code adapted from Paul Quint's hackathon 3
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    batch_size = FLAGS.batch_size
    for epoch in range(FLAGS.max_epoch_num):
      print("################### EPOCH " + str(epoch) + " #####################")
      print("##################################################\n")
      
      # run gradient steps and report mean loss on train data
      ce_vals = []
      conf_mxs = []
      for i in range(valid_count // batch_size):
        batch_data = valid_data[i*batch_size:(i+1)*batch_size, :]
        batch_labels = valid_labels[i*batch_size:(i+1)*batch_size]
        batch_labels = util.white_hot(batch_labels)
        valid_ce, conf_matrix = session.run([sum_cross_entropy, confusion_matrix_op], {input_placeholder: batch_data, labels: batch_labels})
        ce_vals.append(valid_ce)
        conf_mxs.append(conf_matrix)
      avg_valid_ce = sum(ce_vals) / len(ce_vals)
      print('VALID CROSS ENTROPY: ' + str(avg_valid_ce))
      print('VALIDATION CONFUSION MATRIX:')
      print(str(sum(conf_mxs)))
      classification_rate = util.classification_rate(sum(conf_mxs),10)
      print('VALIDATION CLASSIFICATION RATE:' + str(classification_rate))
      
      ce_vals = []
      for i in range(train_count // batch_size):
        batch_data = train_data[i*batch_size:(i+1)*batch_size, :]
        batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
        batch_labels = util.white_hot(batch_labels)
        _, train_ce = session.run([train_op, sum_cross_entropy], {input_placeholder: batch_data, labels: batch_labels})
        ce_vals.append(train_ce)
      avg_train_ce = sum(ce_vals) / len(ce_vals)
      print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
      
      epochs_since_best += 1
      
      if(best_classification_rate < classification_rate): #tracking best
        best_valid_ce = avg_valid_ce
        best_train_ce = avg_train_ce
        best_epoch = epoch
        best_classification_rate = classification_rate
        epochs_since_best = 0
        saver.save(session, "/work/cse496dl/mshanaha/homework_1/homework_1-0")
        print("BEST FOUND")
        
      if(epochs_since_best >= 12): #early stopping
        print("EARLY STOP")
        break
        
      print("\n##################################################")
      
    print(str(best_valid_ce))
    print(str(best_train_ce))
    print(str(best_epoch))

if __name__ == "__main__":
    tf.app.run()
