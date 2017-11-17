##!flask/bin/python3

#
# ===========Guidelines for Usama================
#	comment one # from first line if you are to run it as executable i.e ./app.py
#   otherwise python app.py should be fine.
#	JSON file is just dumped there from ... http://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask
#	Image upload works fine, check the myclassify() function it is the view function
#	I left work incomplete. Pull request me on this one... 

# http://127.0.0.1:5000/classify?query=who+are+you
# http://127.0.0.1:5000/api/v1.0/classifier?q=who+are+you

import os
from flask import Flask , jsonify, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import sys
import json
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'csv', 'json'])

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
# 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#JSON
tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]

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

@app.route('/', methods=['GET', 'POST'])
def welcome():
	return "Classify Kaggle San Francisco Crime Description"

@app.route('/classify', methods=["GET"])
def classify():
    
    # sentence = request.args.get('query')
    # return_list = utility.classify(sentence)
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

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

def load_test_data(test_file, labels):
	df = pd.read_csv(test_file, sep='|')
	select = ['Descript']

	df = df.dropna(axis=0, how='any', subset=select)
	test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

	num_labels = len(labels)
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

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)

	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = './predicted_results_' + timestamp + '/'
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

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
				predictions = sess.run([cnn_rnn.predictions], feed_dict)
				return predictions

			checkpoint_file = trained_dir + 'best_model.ckpt'
			saver = tf.train.Saver(tf.all_variables())
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			logging.critical('{} has been loaded'.format(checkpoint_file))

			batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

			predictions, predict_labels = [], []
			for x_batch in batches:
				batch_predictions = predict_step(x_batch)[0]
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])

			df['PREDICTED'] = predict_labels
			columns = sorted(df.columns, reverse=True)
			df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')

			if y_test is not None:
				y_test = np.array(np.argmax(y_test, axis=1))
				accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
				logging.critical('The prediction accuracy is: {}'.format(accuracy))

			logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

if __name__ == '__main__':
	# python3 predict.py ./trained_results_1478563595/ ./data/small_samples.csv
	# predict_unseen_data()
    app.run(debug=True)