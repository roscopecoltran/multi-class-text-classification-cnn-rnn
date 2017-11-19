#!/usr/bin/env python3
<<<<<<< HEAD
# -*- coding: utf-8 -*-

# report: https://github.com/prabh-me/multi-class-text-classification-cnn-rnn/commit/8c087ec93f105906f0eaedae3f4f9fc4b72865a7

import os
import sys
import json
import time
import datetime
import base64

import ruamel.yaml
=======
import os
import sys
import json
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
<<<<<<< HEAD
import spacy
import tornado.wsgi
import tornado.httpserver

from functools import wraps
import argparse

from flask import Flask, jsonify, request, redirect, url_for, send_from_directory
from flask_cors import CORS, cross_origin
from flask_json import FlaskJSON, JsonError, json_response
# from flask_restplus import Api, Resource, fields, marshal_with, reqparse

from flasgger import Swagger
# from werkzeug.datastructures import FileStorage
from werkzeug import secure_filename
# from flask import Flask, request
# from flask_restful import Resource, Api, reqparse

import tensorflow as tf
# tf.python.control_flow_ops = tf

from text_cnn_rnn import TextCNNRNN
from sklearn.metrics import precision_score, recall_score, f1_score
# from pprint import pprint

# import utils

# from keras.models import load_model

# https://github.com/JoshGlue/RU-TextMining/blob/master/web.py

# from flask.ext.mysqldb import MySQL
# My SQL Server Configurations
# app.config['MYSQL_HOST'] = '127.0.0.1'
# app.config['MYSQL_USER'] = 'test'
# app.config['MYSQL_PASSWORD'] = 'test'
# app.config['MYSQL_DB'] = 'test'
# mysqlInstance = MySQL(app)

# global graph

# Server Parameters
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

def print_tf_flags():
	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
	     print("{}={}".format(attr.upper(), value))
	print("")

# ref. https://github.com/kinni/char-cnn-text-classification-tensorflow/blob/master/serve.py
# print('Loading data')
# x, y, vocabulary, vocabulary_inv = utils.load_data()

"""
Restore the model
"""
# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
# graph = tf.Graph()
# with graph.as_default():
    # sess = tf.Session()
    # with sess.as_default():
        # Load the saved meta graph and restore variables
        # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        # input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        # dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        # predictions = graph.get_operation_by_name("output/predictions").outputs[0]

logging.getLogger().setLevel(logging.INFO)

def parse_postget(f):
    @wraps(f)
    def wrapper(*args, **kw):
        try:
            d = dict((key, request.values.getlist(key) if len(request.values.getlist(
                key)) > 1 else request.values.getlist(key)[0]) for key in request.values.keys())
        except BadRequest as e:
            raise Exception("Payload must be a valid json. {}".format(e))
        return f(d)
    return wrapper

def load_graph(graph_filename):
    with tf.gfile.FastGFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph

=======
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN

logging.getLogger().setLevel(logging.INFO)

>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

<<<<<<< HEAD
def map_word_to_index(examples, words_index):
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

def load_test_data(test_file, labels):
	print("test_file: ", test_file)
	df = pd.read_csv(test_file, sep='|')
	select = ['Description']
=======
def load_test_data(test_file, labels):
	df = pd.read_csv(test_file, sep='|')
	select = ['Descript']
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a

	df = df.dropna(axis=0, how='any', subset=select)
	test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

<<<<<<< HEAD
	# https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
	# count_row=df.shape[0] # gives number of row count
	# count_col=df.shape[1] # gives number of col count
	# total_rows=len(df.axes[0])
	# total_cols=len(df.axes[1])
	# len(df.index) # It's similar.
	# len(df.columns)  
	print("df length: ", len(df['Description']))

=======
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
	num_labels = len(labels)
	logging.info('labels: {}'.format(labels))
	logging.info('num_labels: {}'.format(num_labels))
	
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if 'Category' in df.columns:
		select.append('Category')
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	return test_examples, y_, df

<<<<<<< HEAD
def load_payload_data(payload, labels):
	# print("payload: ", payload)
	# df = pd.read_json(json.loads(payload))
	df = pd.read_json(payload)
	select = ['query']
	df = df.dropna(axis=0, how='any', subset=select)
	test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()
	num_labels = len(labels)
	logging.info('labels: {}'.format(labels))
	logging.info('num_labels: {}'.format(num_labels))
	
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if 'Category' in df.columns:
		select.append('Category')
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	return test_examples, y_, df

def predict_unseen_data():

	### model
	trained_dir = sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	params, words_index, labels, embedding_mat = load_trained_params(trained_dir) # load_trained_params
	
	### input
	x_, y_, df = None, None, None
	print("argv length: ", len(sys.argv))
	if len(sys.argv) > 2:
		test_file = sys.argv[2]
		print("test_file file_path: " , test_file)
		x_, y_, df = load_test_data(test_file, labels)
	else:
		test_payload = '''[
						  {
						    "query": "BURGLARY OF STORE, FORCIBLE ENTRY"
						  }
						]'''
		x_, y_, df = load_payload_data(test_payload, labels)

	# print("[BEFORE] x_: ", x_)
	# print("[BEFORE] y_: ", y_)
	# print("[BEFORE] df: ", df)
	x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
	x_ = map_word_to_index(x_, words_index)
	# print("[AFTER] x_: ", x_)
=======
def map_word_to_index(examples, words_index):
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

def predict_unseen_data():
	trained_dir = sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	test_file = sys.argv[2]

	params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
	x_, y_, df = load_test_data(test_file, labels)
	x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
	x_ = map_word_to_index(x_, words_index)

>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)

<<<<<<< HEAD
	# print("[AFTER] x_test: ", x_test)
	# print("[AFTER] y_test: ", y_test)

	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = '../shared/results/latest/sf-crimes/predict/' # _' + timestamp + '/'
=======
	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = './shared/results/latest/sf-crimes/predicted_' + timestamp + '/'
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

	# refs.
	# - https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
<<<<<<< HEAD
	# - with probs: https://github.com/jamesmw423/multi-class-text-classification-cnn-rnn/commit/33a5883455fd6b75b966de523bcd240d2fefe9ac
=======
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
<<<<<<< HEAD
			# sf-crimes_small.csv, sequence_length = 14
			# payload, sequence_length = 14
			sequence_length = len(x_test[0])
			print("sequence_length: " , sequence_length)
			sys.exit('sequence_length debug')
=======
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
			cnn_rnn = TextCNNRNN(
				embedding_mat = embedding_mat,
				non_static = params['non_static'],
				hidden_unit = params['hidden_unit'],
<<<<<<< HEAD
				sequence_length = params['sequence_length'],
				# sequence_length = len(x_test[0]),
=======
				sequence_length = len(x_test[0]),
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
				max_pool_size = params['max_pool_size'],
				filter_sizes = map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				num_classes = len(labels),
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

<<<<<<< HEAD
			# https://github.com/zackhy/TextClassification/blob/master/cnn_classifier.py
			# 
			# def predict_step(x_batch):

			# https://github.com/tensorflow/tensorflow/issues/97
			# https://github.com/Dong--Jian/Vision-Tutorial/blob/master/code/01-neural-network/net/basic_mlp.py#L36
			def predict_step(x_batch):
				# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
				# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
=======
			def predict_step(x_batch):
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
<<<<<<< HEAD
				# ['input_x', 'input_y', 'dropout_keep_prob', 'batch_size', 'pad', 'real_len', 'embedded_chars', '_initial_state', 'W', 'scores', 'predictions', 'loss', 'accuracy', 'num_correct']
				# print(" #### CNN_RNN KEYS: ", cnn_rnn.__dict__.keys())
				predictions = sess.run([cnn_rnn.predictions], feed_dict)
				probs = (sess.run([cnn_rnn.probs], feed_dict))[0]
				max_probs = [np.max(x) for x in probs]
				return [predictions, max_probs]

			checkpoint_file = trained_dir + 'best_model.ckpt'
			# old: saver = tf.train.Saver(tf.all_variables())
			saver = tf.train.Saver(tf.global_variables())
=======
				predictions = sess.run([cnn_rnn.predictions], feed_dict)
				return predictions

			checkpoint_file = trained_dir + 'best_model.ckpt'
			saver = tf.train.Saver(tf.all_variables())
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			logging.info('{} has been loaded'.format(checkpoint_file))

			batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
<<<<<<< HEAD
			results = []
			predictions, predict_labels, predict_scores = [], [], []
			# https://github.com/pranaypourkar/Blind_Assistive_Device/blob/master/label_image1.py
			# https://github.com/WGierke/git_better/blob/master/app/classifier.py
			# https://github.com/toludwig/golden-lemurs/blob/master/classification/networks/TextCNN.py
			# https://github.com/toludwig/golden-lemurs/blob/master/classification/networks/Ensemble.py
			# https://github.com/toludwig/golden-lemurs/blob/master/classification/networks/LSTM.py
			# https://github.com/toludwig/golden-lemurs/blob/master/classification/networks/NumericFFN.py
			# https://github.com/toludwig/golden-lemurs/blob/master/classification/rate_url.py
			# https://github.com/toludwig/golden-lemurs/blob/master/classification/eval.py
			# https://github.com/toludwig/golden-lemurs/blob/master/classification/networks/Ensemble.py#L33
			# https://github.com/SampannaKahu/qna-classifier/blob/master/code/server.py
			# https://github.com/ematvey/hierarchical-attention-networks/blob/master/worker.py
			# https://github.com/ematvey/hierarchical-attention-networks
			# https://github.com/philipperemy/stock-volatility-google-trends/blob/master/run_model.py
			# https://github.com/metalaman/Text_Classification_Using_CNN/blob/master/Utility.py
			# https://github.com/metalaman/Text_Classification_Using_CNN/blob/master/model.py
			# https://github.com/JoshGlue/RU-TextMining/blob/master/web.py
			# https://github.com/nooralahzadeh/LTG-SIRIUS/blob/master/Project/classifier_markup/CNN_multiLableClassification/eval.py#L99

			for x_batch in batches:
				prediction_output = predict_step(x_batch)
				batch_predictions = prediction_output[0][0]
				max_probs = prediction_output[1]
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])			
					results.append(labels[batch_prediction])
					# predict_scores.append(predict_label_score)
					# print("batch_prediction: ", batch_prediction, "predict_label_idx: ", predict_label_idx,  "predict_label_match: ", labels[predict_label_idx], "predict_label_score: ", predict_label_score)

				for batch_score in max_probs:
					predict_scores.append(batch_score)			

				print("batch_results_count: ", len(batch_predictions))

			print("max_probs length: ", len(max_probs))
			print("batch_predictions length: ", len(batch_predictions))
			print("predict_labels length: ", len(predict_labels))

			# https://stackoverflow.com/questions/42382263/valueerror-length-of-values-does-not-match-length-of-index-pandas-dataframe-u
			# df['B'] = pd.Series([3,4])
			# df.apply(lambda col: col.drop_duplicates().reset_index(drop=True))
			# df['probs'] = pd.Series(max_probs)

			df['Score'] = predict_scores
			df['Match'] = predict_labels			

			columns = sorted(df.columns, reverse=True)

			# to CSV
			df.to_csv('results.csv', index=False, columns=columns, sep='|')
			# df.to_csv(predicted_dir + 'results.csv', index=False, columns=columns, sep='|')

			# Aggregated output
			output = {}
			output["request"] = {} # input parameters
			output["request"]["parameters"] = params			
			output["response"] = {} # output results
			output["response"]["details"] = json.loads(df.to_json(orient = 'records'))
			output["response"]["results"] = results
			output["response"]["log_path"] = predicted_dir			
=======

			predictions, predict_labels = [], []
			for x_batch in batches:
				batch_predictions = predict_step(x_batch)[0]
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])

			df['PREDICTED'] = predict_labels
			columns = sorted(df.columns, reverse=True)
			df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a

			if y_test is not None:
				y_test = np.array(np.argmax(y_test, axis=1))
				accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
<<<<<<< HEAD
				output["response"]["accuracy"] = np.float(accuracy)
				logging.info('The prediction accuracy is: {}'.format(accuracy))

			# to JSON
			with open(predicted_dir + 'results.json', 'wt') as outfile:
				json.dump(output, outfile, indent=4, sort_keys=True)

			# to YAML
			# with open(predicted_dir + 'results.yaml', 'wt') as outfile:
			# 	ruamel.yaml.dump({'results': json.loads(df.to_json(orient='values'))}, outfile, default_flow_style=False)
			with open(predicted_dir + 'results.debug.yaml', 'wt') as outfile:
				ruamel.yaml.dump(output, outfile, default_flow_style=False)
			logging.info('Prediction is complete, all files have been saved: {}'.format(predicted_dir))


def load_graph(graph_filename):
    with tf.gfile.FastGFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph

def load_trained_model(trained_dir, x_test=None):
	### model
	# trained_dir = prefix_path # sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	params, words_index, labels, embedding_mat = load_trained_params(trained_dir) # load_trained_params

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat = embedding_mat,
				non_static = params['non_static'],
				hidden_unit = params['hidden_unit'],				
				sequence_length = 1, # 256
				max_pool_size = params['max_pool_size'],
				filter_sizes = map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				num_classes = len(labels),
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])
			return cnn_rnn, params, words_index, labels, embedding_mat

# 
# https://github.com/benman1/tensorflow_flask/blob/master/api.py
def api():
	app = Flask(__name__, static_url_path='')
	UPLOAD_FOLDER = '../shared/uploads'
	ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'csv', 'json', 'yaml', 'yml'])

	# app.secret_key = 'super_secret_key'
	app.config['RESTPLUS_MASK_SWAGGER'] = False
	app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
	app.config.SWAGGER_UI_DOC_EXPANSION = 'list'

	# app.config['APPLICATION_ROOT'] = '/' + options.api_namespace
    # api = Api(app, version='0.3', title='COMRADES Event API',
    #          description='A set of tools for analysing short textual documents (e.g. tweets).',
    #          doc='/' + options.api_namespace + '/',
    #          endpoint='/' + options.api_namespace
    #          )
    # ns = api.namespace(options.api_namespace,
    #                   description='Event detection tools.')

    # Load models:
    # type_classifier, related_classifier, info_classifier = __load_models()

	CORS(app)
	swagger = Swagger(app)

	cnn_rnn, params, words_index, labels, embedding_mat = load_trained_model(sys.argv[1])

	# https://github.com/evhart/crees/blob/master/crees_server.py#L101

	@app.route('/', methods=['GET', 'POST'])
	@parse_postget
	@cross_origin()
	def get_index():
	    """
	    Index API, returns "Hello!"
	    ---
	    operationId: getPetsById
	    responses:
	      200:
	        description: the word "Hello!"
	    """
	    result = {}
	    result['status'] = 200
	    result['msg'] = "Hello!"
	    return jsonify(result)

	@app.route('/predict', methods=["POST"])
	@parse_postget
	@cross_origin()
	def predict():
	    with graph.as_default():
	        data = request.get_json(force=True)
	        image = read_image(data)
	        number = predict_with_keras(image)
	        return json_response(number=number)

	@app.route('/multiclass', methods=['POST'])
	def predict():
	    if not request.json or not 'text' in request.json:
	        abort(400)

	    text = request.json['text']
	    raw_x = utils.sentence_to_index(text, vocabulary, x.shape[1])
	    predicted_results = sess.run(predictions, {input_x: raw_x, dropout_keep_prob: 1.0})

	    return jsonify({'result': predicted_results[0]})

	@app.route("/classify", methods=['GET'])
	@parse_postget
	@cross_origin()	
	def classify():
		input = request.args.get('content')
		result = {}
		result['status'] = 200
		result['msg'] = input
		return jsonify(result)

	def real_len(batches):
		return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

	def predict_step(x_batch):
		feed_dict = {
			cnn_rnn.input_x: x_batch,
			cnn_rnn.dropout_keep_prob: 1.0,
			cnn_rnn.batch_size: len(x_batch),
			cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
			cnn_rnn.real_len: real_len(x_batch),
		}
		# ['input_x', 'input_y', 'dropout_keep_prob', 'batch_size', 'pad', 'real_len', 'embedded_chars', '_initial_state', 'W', 'scores', 'predictions', 'loss', 'accuracy', 'num_correct']
		predictions = sess.run([cnn_rnn.predictions], feed_dict)
		return predictions

	# ref. https://github.com/PrasenjitGiri/TensorflowFlaskApp/blob/master/server.py
    # tf_graph = load_graph('output_graph.pb')
    # softmax_tensor = tf_graph.get_tensor_by_name('final_result:0')
    # label_lines = [line.strip() for line in tf.gfile.GFile("output_labels.txt")]
    # persistent_session = tf.Session(graph=tf_graph)

    # model = load_model('/model/final_model.h5')
    # Store graph to allow inference in a different thread:
    # https://github.com/fchollet/keras/issues/2397
    # graph = tf.get_default_graph()

	# server_port = int(os.getenv('SERVER_PORT', 8000))
	app.run(host='0.0.0.0', port=8000, threaded=True, debug=True)

# celery: https://github.com/Sil2204/Tensorflow-celery-flask-dialogue/blob/master/example_server_thread.py
# uwsgi: https://github.com/pkmital/flask-uwsgi-tensorflow
# docker: https://github.com/Vetal1977/tf_serving_flask_app/blob/master/docker-compose.yaml
if __name__ == '__main__':
	# api()
=======
				logging.critical('The prediction accuracy is: {}'.format(accuracy))

			logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

if __name__ == '__main__':
	# python3 predict.py ./shared/results/latest/sf-crimes/trained ./shared/data/sf-crimes/dataset/small_samples.csv
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
	predict_unseen_data()
