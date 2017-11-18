# Imports for data io operations
from collections import deque
from six import next
import readers
import argparse

# Main imports for training
import tensorflow as tf
import numpy as np

# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(42)

parser = argparse.ArgumentParser(description='Reccomandation engine using svd')

parser.add_argument('--data', action="store", dest="score_file", default='../data/ratings.dat')
parser.add_argument('--model', action="store", dest="mode_dir", default='../save/')
parser.add_argument('--seprater', action="store", dest="seprater", default='::');
parser.add_argument('--batch_size', action="store", dest="batch_size",default=1000,type=int)
parser.add_argument('--dims', action="store", dest="dims",default=15, type=int)
parser.add_argument('--epochs', action="store", dest="max_epochs",default=25, type=int)
parser.add_argument('--device', action="store", dest="place_device",default='/cpu:0')
parser.add_argument('--item', action="store", dest="item",default=3952 , type=int)
parser.add_argument('--user', action="store", dest="user",default=6040  , type=int)
parser.add_argument('--learning_rate', action="store", dest="learning_rate",default=0.01  , type=int)
parser.add_argument('--exp', action="store", dest="exp_id",default='learning_rate_0.001_15')


arg = parser.parse_args()

seprater = arg.seprater
batch_size = arg.batch_size # Number of samples per batch
dims = arg.dims         # Dimensions of the data, 15
max_epochs = arg.max_epochs   # Number of times the network sees all the training data
learning_rate = arg.learning_rate
exp_id = arg.exp_id

u_num = arg.user # Number of users in the dataset
i_num = arg.item  # Number of movies in the dataset

# Device used for all computations
place_device = arg.place_device

mode_dir = arg.mode_dir

def get_data(score_file,seprater):
    # Reads file using the demiliter :: form the ratings file
    # Download movie lens data from: http://files.grouplens.org/datasets/movielens/ml-1m.zip
    # Columns are user ID, item ID, rating, and timestamp
    # Sample data - 3::1196::4::978297539
    df = readers.read_file(score_file, sep=seprater)
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

def model(user_batch, item_batch, user_num, item_num, dim, device):
    with tf.device(device):
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
    return infer, regularizer, w_user, w_item

def loss(infer, regularizer, rate_batch, learning_rate, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        # Use L2 loss to compute penalty
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        # Reference: http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
        train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
    return cost, train_op

# Read data from ratings file to build a TF model
df_train, df_test = get_data(arg.score_file,seprater)
del df_train


# Sequentially generate one-epoch batches, for testing
iter_test = readers.OneEpochIterator([df_test["user"],
                                     df_test["item"],
                                     df_test["rate"]],
                                     batch_size=-1)

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer,w_user,w_item = model(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dims, device=place_device)
_, train_op = loss(infer, regularizer, rate_batch, learning_rate=0.01, reg=0.05, device=place_device)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    new_saver = tf.train.import_meta_graph(mode_dir+exp_id+'.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(mode_dir))
    test_err2 = np.array([])
    for users, items, rates in iter_test:
        pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                item_batch: items})
        pred_batch = clip(pred_batch)
        print("Pred\tActual")
        for ii in range(10):
            print("%.3f\t%.3f" % (pred_batch[ii], rates[ii]))
        test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
        print(np.sqrt(np.mean(test_err2)))