'''
Distributed Tensorflow example of using data parallelism and share model parameters.
Trains a simple reccomandation engine using Distributed Tensorflow

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python tensorflow-bot.py --job-name="ps" --task_index=0 
pc-02$ python tensorflow-bot.py --job-name="worker" --task_index=0 
pc-03$ python tensorflow-bot.py --job-name="worker" --task_index=1 

More details here: evalsocket.com
'''

from __future__ import print_function

import tensorflow as tf
from collections import deque
from six import next
import readers
import sys
import time

# cluster specification
parameter_servers = ["pc-01:2222"]
workers = [ "pc-02:2222", 
      "pc-03:2222"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

# config
batch_size = 1000
learning_rate = 0.1

u_num = 6040 # Number of users in the dataset
i_num = 3952 # Number of movies in the dataset

dims = 15         # Dimensions of the data, 15
max_epochs = 25 

logs_path = "../save/"


def get_data():
    # Reads file using the demiliter :: form the ratings file
    # Download movie lens data from: http://files.grouplens.org/datasets/movielens/ml-1m.zip
    # Columns are user ID, item ID, rating, and timestamp
    # Sample data - 3::1196::4::978297539
    df = readers.read_file("../data/ratings.dat", sep="::")
    rows = len(df)
    # Purely integer-location based indexing for selection by position
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    # Separate data into train and test, 90% for train and 10% for test
    split_index = int(rows * 0.9)
    # Use indices to separate the data
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    
    return df_train, df_test

def clip(x):
    return np.clip(x, 1.0, 5.0)
  
def model(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0",parameter="/cpu:0"):
    with tf.device(parameter):
        # Using a global bias term
        bias_global = tf.get_variable("bias_global", shape=[])
        # User and item bias variables
        # get_variable: Prefixes the name with the current variable scope 
        # and performs reuse checks.
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        # embedding_lookup: Looks up 'ids' in a list of embedding tensors
        # Bias embeddings for user and items, given a batch
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        # User and item weight variables
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        # Weight embeddings for user and items, given a batch
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    
    with tf.device(device):
        # reduce_sum: Computes the sum of elements across dimensions of a tensor
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        # l2_loss: Computes half the L2 norm of a tensor without the sqrt
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), 
                             name="svd_regularizer")
    return infer, regularizer
  
def loss(infer, regularizer, rate_batch, learning_rate=0.1, reg=0.1, device="/cpu:0",parameter="/cpu:0"):
    with tf.device(device):
        # Use L2 loss to compute penalty
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
    return cost, train_op

def eval_session(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=10, device="/cpu:0",parameter="/cpu:0"){
   infer, regularizer = model(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dim, device,parameter)
   _, train_op = loss(infer, regularizer, rate_batch, learning_rate=0.10, reg=0.05, device,parameter)
   return infer,regularizer,train_op,_
 }

if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":
  
  # load data set
  df_train, df_test = get_data()
  parameter_device="/job:ps/task:%d" % FLAGS.parameter_index;
  samples_per_batch = len(df_train) // batch_size
  print("Number of train samples %d, test samples %d, samples per batch %d" % 
      (len(df_train), len(df_test), samples_per_batch))
  
  # Using a shuffle iterator to generate random batches, for training
  iter_train = readers.ShuffleIterator([df_train["user"],
                                       df_train["item"],
                                       df_train["rate"]],
                                       batch_size=batch_size)

  # Sequentially generate one-epoch batches, for testing
  iter_test = readers.OneEpochIterator([df_test["user"],
                                       df_test["item"],
                                       df_test["rate"]],
                                       batch_size=-1)
  
  with tf.device(parameter_device):
    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None],name="score")
    
  # Master Node Setup
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:0",
    cluster=cluster)):
      if FLAGS.job_type == "train":
        worker_device="/job:worker/task:%d" % FLAGS.train_index;
        infer,regularizer,train_op,_ = eval_session(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dims, device=worker_device,parameter=parameter_device)
        
      elif FLAGS.job_type == "test":
        worker_device="/job:worker/task:%d" % FLAGS.test_index;
        infer,regularizer,train_op,_ = eval_session(infer, regularizer, rate_batch, learning_rate=0.10, reg=0.05, device=worker_device,parameter=parameter_device)
      
      # Inference using saved model
      init_op = tf.global_variables_initializer()

      # specify optimizer
      with tf.name_scope('session start'):
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
          sess.run(init_op)
          print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
          for i in range(max_epochs * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                               item_batch: items,
                                                               rate_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()

                print("%02d\t%.3f\t\t%.3f\t\t%.3f secs" % (i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)), end - start))
                start = end
          saver.save(sess, '../save/')