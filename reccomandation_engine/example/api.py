# Imports for data io operations
from collections import deque
from six import next
from flask import Flask, jsonify, render_template, request
import readers
import argparse

# webapp
app = Flask(__name__)

# Main imports for training
import tensorflow as tf
import numpy as np

# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(42)

parser = argparse.ArgumentParser(description='Reccomandation engine using svd')

parser.add_argument('--data', action="store", dest="score_file", default='../data/ratings.dat')
parser.add_argument('--model', action="store", dest="mode_dir", default='../save')
parser.add_argument('--seprater', action="store", dest="seprater", default='::');
parser.add_argument('--batch_size', action="store", dest="batch_size",default=1000,type=int)
parser.add_argument('--dims', action="store", dest="dims",default=15, type=int)
parser.add_argument('--epochs', action="store", dest="max_epochs",default=25, type=int)
parser.add_argument('--device', action="store", dest="place_device",default='/cpu:0')
parser.add_argument('--item', action="store", dest="item",default=3952 , type=int)
parser.add_argument('--user', action="store", dest="user",default=6040  , type=int)
parser.add_argument('--learning_rate', action="store", dest="learning_rate",default=0.001  , type=int)
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
    return infer, regularizer,w_user,w_item

def loss(infer, regularizer, rate_batch, learning_rate, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        # Use L2 loss to compute penalty
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        # Reference: http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
        train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
    return cost, train_op

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer,w_user,w_item = model(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dims, device=place_device)
_, train_op = loss(infer, regularizer, rate_batch, learning_rate=0.01, reg=0.05, device=place_device)

saver = tf.train.Saver()

# Inference using saved model
init_op = tf.global_variables_initializer()

dir_path = mode_dir + '/' +exp_id

sess = tf.Session();

def predict(user,item):
            sess.run(init_op)
            new_saver = tf.train.import_meta_graph(dir_path+'.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(mode_dir))
            pred_batch = sess.run(infer, feed_dict={user_batch: user,
                                                item_batch: item})
            pred_batch = clip(pred_batch)
            return pred_batch;
          
def user_distance(user):
            sess.run(init_op)
            new_saver = tf.train.import_meta_graph(dir_path+'.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(mode_dir))
            w = sess.run(w_user);
            distance = tf.reduce_sum(tf.abs(tf.add(w, tf.negative(w[user]))), reduction_indices=1)
            top_k, top_k_indices = tf.nn.top_k(tf.negative(distance), k=10)
            pred = sess.run(top_k_indices)
            return pred;
          
def item_distance(item):
            sess.run(init_op)
            new_saver = tf.train.import_meta_graph(dir_path+'.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(mode_dir))
            w = sess.run(w_item);
            distance = tf.reduce_sum(tf.abs(tf.add(w, tf.negative(w[item]))), reduction_indices=1)
            top_k, top_k_indices = tf.nn.top_k(tf.negative(distance), k=10)
            pred = sess.run(top_k_indices)
            return pred;

@app.route('/api/predict_rating', methods=['GET'])
def predict_rating():
    user = np.array([int(request.args.get('user'))])
    item = np.array([int(request.args.get('item'))])
    output = predict(user,item);
    return jsonify(result={"rating":output.tolist()})
  
@app.route('/api/simillar_user', methods=['GET'])
def simillar_user():
    user = np.array([int(request.args.get('user'))])
    output = user_distance(user);
    return jsonify(result={"simillar_user":output.tolist()})

@app.route('/api/simillar_item', methods=['GET'])
def simillar_item():
    item = np.array([int(request.args.get('item'))])
    output = item_distance(item);
    return jsonify(result={"simillar_item":output.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)