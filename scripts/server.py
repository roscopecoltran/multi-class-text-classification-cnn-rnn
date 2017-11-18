#!/usr/bin/env python3
import os
import sys
import json
import time
import datetime

import ruamel.yaml
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import spacy

import argparse
from flask import Flask, jsonify, request, redirect, url_for, send_from_directory
# from flask_restful import Resource, Api, reqparse
from werkzeug import secure_filename

import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
from sklearn.metrics import precision_score, recall_score, f1_score
# from pprint import pprint

# https://github.com/JoshGlue/RU-TextMining/blob/master/web.py
# from flask.ext.mysqldb import MySQL
# My SQL Server Configurations
# app.config['MYSQL_HOST'] = '127.0.0.1'
# app.config['MYSQL_USER'] = 'test'
# app.config['MYSQL_PASSWORD'] = 'test'
# app.config['MYSQL_DB'] = 'test'
# mysqlInstance = MySQL(app)

logging.getLogger().setLevel(logging.INFO)

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)

	return params, words_index, labels, embedding_mat

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

def web(cnn_rnn, sess):
    app = Flask(__name__)
	@app.route('/', methods=['GET', 'POST'])
	def welcome():
		# input = request.args.get('text')	
		return "Classify Kaggle San Francisco Crime Description"
    @app.route("/classify", methods=['GET'])
	def classify():
		# input = request.args.get('text')	
        return "classify GET!"
    @app.route("/classify", methods=['POST'])
	def classify():
        return "classify POST!"

def load_model(trained_dir="./models"):
	# trained_dir = sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
	return params, words_index, labels, embedding_mat

def test_model(input_content="", input_type='payload', labels=[]):
	x_, y_, df = None, None, None
	if input_type == "file" && input_content != "":
		x_, y_, df = load_test_data(input_content, labels)
	elif input_type == "payload" && input_content != "":
		x_, y_, df = load_payload_data(input_content, labels)
	else:
		test_payload = '''[
						  {
						    "query": "BURGLARY OF STORE, FORCIBLE ENTRY"
						  }
						]'''
		x_, y_, df = load_payload_data(test_payload, labels)
	return x_, y_, df

def predict_process(x_batch):
	checkpoint_file = trained_dir + 'best_model.ckpt'
	saver = tf.train.Saver(tf.all_variables())
	saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
	saver.restore(sess, checkpoint_file)
	logging.info('{} has been loaded'.format(checkpoint_file))

	batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
	results = []
	predictions, predict_labels = [], []

	for x_batch in batches:
		batch_results = predict_step(x_batch)[0] # [0] # limit to last 3
		print("batch_results_count: ", len(batch_results))
		for label_key in batch_results:
			label_val = labels[label_key]
			predictions.append(label_key)
			predict_labels.append(label_val)
			results.append(label_val)
			print("label_key: ", label_key, "label: ", label_val)

	df['predicted'] = predict_labels
	columns = sorted(df.columns, reverse=True)

	# to CSV
	df.to_csv(predicted_dir + 'results.csv', index=False, columns=columns, sep='|')

	# Aggregated output
	output = {}
	output["request"] = {} # input parameters
	output["request"]["parameters"] = params			
	output["response"] = {} # output results
	output["response"]["details"] = json.loads(df.to_json(orient = 'records'))
	output["response"]["results"] = results
	output["response"]["log_path"] = predicted_dir			

	if y_test is not None:
		y_test = np.array(np.argmax(y_test, axis=1))
		accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
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

def log_predict_session(with_timestamp=False)
	timestamp_suffix = ""
	if with_timestamp is True:
		timestamp_suffix = trained_dir.split('/')[-2].split('_')[-1]+ '/'
	predict_session_dir = '../shared/results/latest/sf-crimes/predict/'+timestamp_suffix
	if os.path.exists(predict_session_dir):
		shutil.rmtree(predict_session_dir)
	os.makedirs(predict_session_dir)

# load tensorflow model
model_prefix_path = ""
if len(sys.argv[1]) < 1:
	sys.exit('Please provide a model as the first argument.')
else:
	model_prefix_path = sys.argv[1]

params, words_index, labels, embedding_mat = load_model(model_prefix_path)
x_, y_, df = test_model()
x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
x_ = map_word_to_index(x_, words_index)

x_test, y_test = np.asarray(x_), None
if y_ is not None:
	y_test = np.asarray(y_)

log_predict_session()

# refs.
# - https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn_rnn = TextCNNRNN(
			embedding_mat = embedding_mat,
			non_static = params['non_static'],
			hidden_unit = params['hidden_unit'],
			sequence_length = len(x_test[0]),
			max_pool_size = params['max_pool_size'],
			filter_sizes = map(int, params['filter_sizes'].split(",")),
			num_filters = params['num_filters'],
			num_classes = len(labels),
			embedding_size = params['embedding_dim'],
			l2_reg_lambda = params['l2_reg_lambda'])

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
			predictions = sess.run([cnn_rnn.predictions], feed_dict) # , cnn_rnn.scores, cnn_rnn.accuracy
			return predictions

		web(cnn_rnn, sess)

#if __name__ == '__main__':
#	predict_unseen_data()