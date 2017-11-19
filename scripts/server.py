#!/usr/bin/env python3
<<<<<<< HEAD
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
=======

# urls:
# - http://127.0.0.1:5000/classify?query=who+are+you
# - http://127.0.0.1:5000/api/v1.0/classifier?q=who+are+you

# refs:
# https://github.com/kashyapakshay/BuzzKill/blob/master/server.py
# https://github.com/eagle705/Explain_DeepLearning_LIME/blob/master/index.py
# https://github.com/eagle705/Explain_DeepLearning_LIME/blob/master/RNN_Flask.py
# https://github.com/eagle705/Explain_DeepLearning_LIME/blob/master/CNN_Flask.py
# https://github.com/RRisto/krattbot/blob/master/flask_app.py
# https://github.com/bogdanned/charles/blob/master/app_platon/app/index.py
# https://github.com/scottming/EmbeddingCNNintoWeb/blob/master/py/app.py
# https://github.com/eBay/Sequence-Semantic-Embedding
# https://github.com/PrasenjitGiri/TensorflowFlaskApp/blob/master/server.py
# https://github.com/SeldonIO/seldon-server/blob/master/python/examples/doc_similarity_api.ipynb
# https://github.com/liuzqt/keyword_spotting/blob/master/server_demo.py
# https://github.com/Top-Ranger/ClassifyHub/blob/master/configserver.py
# https://github.com/scottming/EmbeddingCNNintoWeb/tree/master/py
# https://github.com/vontell/Judgd/blob/master/scraper.py
# https://github.com/cahya-wirawan/text-classification/blob/master/textclassification.py
# https://github.com/Ichaelus/Github-Classifier/blob/master/Playground/nn.py
# https://github.com/Ichaelus/Github-Classifier/blob/master/Application/start.py
# https://github.com/linkvt/repo-classifier
# https://github.com/eamonnmag/magpie_web/blob/master/magpie_web.py

# tf api refs:
# https://github.com/aliostad/wiki-album-genre/blob/8fed753154b3e3744811d0c9eaf0ce1429d89289/album_genre_api.py#L79
# https://github.com/aiteamwvu/PerfLearner_ML/blob/master/NNtestserver.py
# 

# commands:
# $ python3 server.py ./shared/results/latest/sf-crimes/trained_results_1478563595/ ./shared/data/sf-crimes/dataset/small_samples.csv
# $ server.py ./shared/results/latest/sf-crimes/trained_results_1478563595/ ./shared/data/sf-crimes/dataset/small_samples.csv

import os
import sys
import json
import shutil
import pickle
import logging

# from scipy.fftpack import dct
# from flask_socketio import SocketIO
# from flask import Flask, render_template, request,json, Response

import argparse
from flask import Flask , jsonify, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import data_helper

import numpy as np
import pandas as pd
import tensorflow as tf

from text_cnn_rnn import TextCNNRNN

UPLOAD_FOLDER = './shared/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'tsv','csv', 'json', 'yaml', 'yml', 'toml', 'ini'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument("foo", ..., required=True)
# parser.parse_args()
for arg in sys.argv:
    print(" !!! arg=", arg)

# def predict_unseen_data():
trained_dir = sys.argv[1]

if trained_dir == "":
	trained_dir = './shared/results/latest/sf-crimes/trained_results/'

if not trained_dir.endswith('/'):
	trained_dir += '/'

if not os.path.exists(trained_dir):
	logging.info('trained_dir doesn\'t exist: {}'.format(trained_dir))
	sys.exit('Could not find the model specified at: '+ trained_dir)

test_file = ""
if len(sys.argv) > 2:
	test_file = sys.argv[2]

if test_file == "":
	test_file = "./shared/data/sf-crimes/dataset/small_samples.csv"

# if not os.path.isfile(test_file)
#	sys.exit('Could not find the test file expected at: '+ test_file)
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())
<<<<<<< HEAD

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)

	return params, words_index, labels, embedding_mat

=======
	# if not os.path.isfile(trained_dir + 'embeddings.pickle')
	# 	sys.exit('Could not find the file expected at: '+ trained_dir + 'embeddings.pickle')
	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

#def load_test_data(test_file, labels):
	# logging.info('test_file: {}'.format(test_file))
	# df = pd.read_csv(test_file, sep='|')
	# select = ['Description']

	# df = df.dropna(axis=0, how='any', subset=select)
	# test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

	# num_labels = len(labels)
	# one_hot = np.zeros((num_labels, num_labels), int)
	# np.fill_diagonal(one_hot, 1)
	# label_dict = dict(zip(labels, one_hot))

	# y_ = None
	# if 'Category' in df.columns:
	# 	select.append('Category')
	# 	y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

	# not_select = list(set(df.columns) - set(select))
	# df = df.drop(not_select, axis=1)
	# logging.info('test_examples: {}'.format(test_examples))
	# logging.info('y_: {}'.format(y_))
	# logging.info('df: {}'.format(df))
	# return test_examples, y_, df

>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
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

<<<<<<< HEAD
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
	# ref. https://github.com/gallupliu/text_matching_tensorflow/blob/master/multi-class-text-classification-cnn-rnn/report.txt 
	# old: saver = tf.train.Saver(tf.all_variables())
	saver = tf.train.Saver(tf.global_variables())
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
=======
params, words_index, labels, embedding_mat = load_trained_params(trained_dir)

# x_, y_, df = load_test_data(test_file, labels)
# x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
# logging.info('x_ (data_helper): {}'.format(x_))
# logging.info('params[\'sequence_length\']: {}'.format(params['sequence_length']))
# x_ = map_word_to_index(x_, words_index)
# logging.info('x_ (map_word_to_index): {}'.format(x_))
# logging.info('labels: {}'.format(labels))
# logging.info('words_index: {}'.format(words_index))

# numpy array
# x_test, y_test = np.asarray(x_), None
# if y_ is not None:
# 	y_test = np.asarray(y_)

# timestamp = trained_dir.split('/')[-2].split('_')[-1]
# predicted_dir = './shared/results/latest/sf-crimes/predicted_results/'

# prediction output dir
# if os.path.exists(predicted_dir):
# 	logging.info('rmtree previous predicted_dir: {}'.format(predicted_dir))
# 	shutil.rmtree(predicted_dir)

# if not os.path.exists(predicted_dir):
# 	logging.info('makedirs for predicted_dir: {}'.format(predicted_dir))
# 	os.makedirs(predicted_dir)

# logging loaded variables for batch process
# logging.info('timestamp: {}'.format(timestamp))
# logging.info('predicted_dir: {}'.format(predicted_dir))
# logging.info('trained_dir: {}'.format(trained_dir))
# logging.info('test_file: {}'.format(test_file))

# load tensorflow model
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)
	with sess.as_default():		
		# logging.info('x_test[0]: {}'.format(x_test[0]))
		# logging.info('len(x_test[0]): {}'.format(len(x_test[0])))
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
		cnn_rnn = TextCNNRNN(
			embedding_mat = embedding_mat,
			non_static = params['non_static'],
			hidden_unit = params['hidden_unit'],
<<<<<<< HEAD
			sequence_length = len(x_test[0]),
=======
			sequence_length = 5,
			# sequence_length = len(x_test[0]),
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
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
<<<<<<< HEAD
			predictions = sess.run([cnn_rnn.predictions], feed_dict) # , cnn_rnn.scores, cnn_rnn.accuracy
			return predictions

		web(cnn_rnn, sess)

#if __name__ == '__main__':
#	predict_unseen_data()
=======
			predictions = sess.run([cnn_rnn.predictions], feed_dict)
			return predictions

		# restore tensorflow graph
		checkpoint_file = trained_dir + 'best_model.ckpt'
		logging.info('checkpoint_file: {}'.format(checkpoint_file))
		saver = tf.train.Saver(tf.all_variables())
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)
		logging.info('{} has been loaded'.format(checkpoint_file))

		# logging.info('x_test: {}'.format(x_test))
		# batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
		# predictions, predict_labels = [], []

		# batch_predictions = predict_step(x_batch)[0]
		# for x_batch in batches:
			# logging.info('x_batch: {}'.format(x_batch))
			# batch_predictions = predict_step(x_batch)[0]
			# logging.info('batch_predictions: {}'.format(batch_predictions))
			# for batch_prediction in batch_predictions:
			# 	predictions.append(batch_prediction)
			# 	predict_labels.append(labels[batch_prediction])

		# df['PREDICTED'] = predict_labels
		# logging.info('Output: {}'.format(df))
		# columns = sorted(df.columns, reverse=True)
		# df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')

		# logging.info('y_test: {}'.format(y_test))
		#if y_test is not None:
		#	y_test = np.array(np.argmax(y_test, axis=1))
		#	accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
		#	logging.critical('The prediction accuracy is: {}'.format(accuracy))

		# logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

def test_sentence(sentence):
    sentence = process_text(sentence)
    vec = vectorizer.transform([sentence]).toarray()
    print("Label", label_names[np.argmax(model.predict_proba(vec))])
    out = model.predict_proba(vec)[0]
    prob_dev = out[0]
    prob_not = out[1]
    print("Proba DEV: {0:.2f} Not DEV: {1:.2f}".format(prob_dev, prob_not))

# Utility functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def my_classify(uploadfolder, filename):
	'''
	this is called a View function in FLASK terminology
	it must return something
	'''
	return "hello"


#@app.route('/text', methods=['POST'])
#def send():
#    name = request.form['text']
#    conn = mongo_conn()
#    result = conn.send_data(name)
#    return render_template('hello.html', result = result)

@app.route('/', methods=['GET', 'POST'])
def welcome():
	return "Classify Kaggle San Francisco Crime Description"

@app.route('/classify', methods=["GET"])
def classify():
    sentence = request.args.get('query')
    return_list = utility.classify(sentence)
    return return_list

@app.route('/api/v1.0/classifier', methods=['GET', 'POST'])
def upload_file():
    '''
    Possible Exposing of Classify method here?????
    request method is POST.. so send image here.
    notice ELSE part sends the form with action POST.
    '''
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return my_classify(app.config['UPLOAD_FOLDER'] , filename)
            #return redirect(url_for('uploaded_file',filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <h3>TESTING FILE UPLOAD </h3>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>'''

#This method returns what happens after the file is uploaded
@app.route('/api/v1.0/classifier/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
    #This is sent when BASE URL recieves a POST or GET

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> 301605969ead13f36611301c9a96ef1cbaa8477a
