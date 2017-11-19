#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ref. https://github.com/JoshGlue/RU-TextMining/blob/master/web.py
# report: https://github.com/prabh-me/multi-class-text-classification-cnn-rnn/commit/8c087ec93f105906f0eaedae3f4f9fc4b72865a7
# run: cd scripts && python3 web.py ../shared/results/latest/sf-crimes/trained_results_1510012745

import os
import sys
import json
import time
import datetime
import base64
import ruamel.yaml
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
from functools import wraps
import argparse

# import tornado.wsgi
# import tornado.httpserver
from flask import Flask, jsonify, request, redirect, url_for, send_from_directory, g, render_template
from flask_cors import CORS, cross_origin
from flask_json import FlaskJSON, JsonError, json_response

# from flask_restplus import Api, Resource, fields, marshal_with, reqparse
from flasgger import Swagger
from werkzeug import secure_filename

# import spacy
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
from sklearn.metrics import precision_score, recall_score, f1_score

logging.getLogger().setLevel(logging.INFO)

app = Flask(__name__)
author = "M.L."

def print_tf_flags():
	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
	     print("{}={}".format(attr.upper(), value))
	print("")

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

def web(model, sess):

	# Tensorflow model
	sess.run(tf.initialize_all_variables())
	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = '../shared/results/latest/sf-crimes/predict/' # _' + timestamp + '/'
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

	checkpoint_file = trained_dir + 'best_model.ckpt'
	saver = tf.train.Saver(tf.global_variables())
	saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
	saver.restore(sess, checkpoint_file)
	logging.info('{} has been loaded'.format(checkpoint_file))

	# api service
	app = Flask(__name__)
	# app = Flask(__name__, static_url_path='')
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
	CORS(app)
	swagger = Swagger(app)

	headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

	@app.before_request
	def before_request():
		### ref. https://gist.github.com/lost-theory/4521102
		# request_start_time = time.time()
		# g.request_time = lambda: "%.5fs" % (time.time() - request_start_time)
	    g.request_start_time = time.time()
	    g.request_time = lambda: "%.5fs" % (time.time() - g.request_start_time)

	@app.route('/', methods=['GET'])
	# @parse_postget
	@cross_origin()
	def get_index():
	    result = {}
	    result['status'] = 200
	    result['msg'] = "Hello!"
	    return jsonify(result)

	@app.route('/predict', methods=['POST'])
	def predict():
	    if not request.json or not 'content' in request.json:
	        abort(400)
	    content = request.json['content']
	    result = {}
	    result['status'] = 200
	    result['msg'] = content
	    return jsonify(result)

	@app.route('/extract-feedback', methods=['POST'])
	def extract_feedback():
	    print request.form
	    text = request.form.get('text', '')
	    return redirect('/thanks')

	@app.route('/word2vec', methods=['POST', 'GET'])
	def word2vec():
	    if request.method == 'POST':
	        positive = request.form.get('positive', None)
	        negative = request.form.get('negative', None)

	        data = {'corpus': 'keywords'}
	        ctx = {'type': 'word2vec'}
	        if positive:
	            data['positive'] = [w.strip() for w in positive.split(',')]
	            ctx['positive'] = ", ".join(data['positive'])
	        if negative:
	            data['negative'] = [w.strip() for w in negative.split(',')]
	            ctx['negative'] = ", ".join(data['negative'])

	        response = requests.post(WORD2VEC_URL,
	                                 data=json.dumps(data),
	                                 headers=headers)
	        contents = json.loads(response.text)

	        return render_template('brain/results.html', results=contents, ctx=ctx)
	    else:
	        return render_template('brain/word2vec.html')

	@app.route('/thanks', methods=['GET'])
	def thanks():
	    return render_template('brain/thanks.html')

	# @app.route("/classify", methods=['GET', 'POST'])
	@app.route("/classify", methods=['GET'])
	# @parse_postget
	@cross_origin()
	def return_score():
		# to do POST, 
		#if not request.json or not 'content' in request.json:
		#    abort(400)
		content = request.args.get('content')
		payload = '[{"query": "' + content + '"}]'

		x_, y_, df = load_payload_data(payload, labels)
		x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
		x_ = map_word_to_index(x_, words_index)
		x_test, y_test = np.asarray(x_), None
		if y_ is not None:
			y_test = np.asarray(y_)

		batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
		predictions, predict_labels, predict_scores = [], [], []

		for x_batch in batches:
			prediction_output = predict_step(x_batch)
			batch_predictions = prediction_output[0][0]
			max_probs = prediction_output[1]
			for batch_prediction in batch_predictions:
				predictions.append(batch_prediction)
				predict_labels.append(labels[batch_prediction])
			for batch_score in max_probs:
				predict_scores.append(batch_score)

		df['score'] = predict_scores
		df['match'] = predict_labels			
		columns = sorted(df.columns, reverse=True)

		output = {}
		output["request"] = {}
		output["request"]["params"] 		= params			
		output["request"]["content"] 		= content
		output["request"]["payload"] 		= payload			

		output["response"] = {}
		output["response"]["results"] 		= json.loads(df.to_json(orient = 'records'))
		output["response"]['response_time'] = g.request_time()

		if y_test is not None:
			y_test = np.array(np.argmax(y_test, axis=1))
			accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
			output["response"]["accuracy"] 	= np.float(accuracy)

		result = {}
		result['status'] = 200
		result['output'] = output
		return jsonify(result)

	def real_len(batches):
		return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

	def predict_step(x_batch):
		feed_dict = {
			model.input_x: x_batch,
			model.dropout_keep_prob: 1.0,
			model.batch_size: len(x_batch),
			model.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
			model.real_len: real_len(x_batch),
		}
		predictions = sess.run([model.predictions], feed_dict)
		probs = (sess.run([model.probs], feed_dict))[0]
		max_probs = [np.max(x) for x in probs]
		return [predictions, max_probs]

	def load_payload_data(payload, labels):
		print("payload: ", payload)
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

    # app.run()
	# server_port = int(os.getenv('SERVER_PORT', 8000))
	# app.run(use_debugger=True, use_reloader=True)
	app.run(host='0.0.0.0', port=8000, threaded=True, debug=True)

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

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())
	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

# load model
trained_dir = sys.argv[1]
if not trained_dir.endswith('/'):
	trained_dir += '/'
params, words_index, labels, embedding_mat = load_trained_params(trained_dir) # load_trained_params

# load graph
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		model = TextCNNRNN(
			embedding_mat = embedding_mat,
			non_static = params['non_static'],
			hidden_unit = params['hidden_unit'],				
			sequence_length = params['sequence_length'],
			max_pool_size = params['max_pool_size'],
			filter_sizes = map(int, params['filter_sizes'].split(",")),
			num_filters = params['num_filters'],
			num_classes = len(labels),
			embedding_size = params['embedding_dim'],
			l2_reg_lambda = params['l2_reg_lambda'])
		web(model, sess)

