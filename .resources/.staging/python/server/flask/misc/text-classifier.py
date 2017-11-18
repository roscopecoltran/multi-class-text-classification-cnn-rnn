#!/usr/local/bin/python3
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import tensorflow as tf
from functools import wraps
from flask import Flask, request, jsonify,render_template
import pickle


"""
Load a tensorflow model and make it available as a REST service
"""
app = Flask(__name__)

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1498107052/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

f = open('./y_target.pickle', 'rb')
lb = pickle.load(f)

@app.route('/',methods=['POST','GET'])
def login():
    return render_template('index.html')


def parse_postget(f):
    @wraps(f)
    def wrapper(*args, **kw):
        try:
            d = dict((key, request.values.getlist(key) if len(request.values.getlist(
                key)) > 1 else request.values.getlist(key)[0]) for key in request.values.keys())
        except:
            raise Exception("Invalid json.")
        return f(d)
    return wrapper

@app.route('/api', methods=['GET', 'POST'])
@parse_postget
def apply_model(data):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            x_raw = [data['input']]

            print x_raw

            print "~~~~~~~~~~~~~~~~~~~~~~~~~~~"


            vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
            vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
            x_test = np.array(list(vocab_processor.transform(x_raw)))


            checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)


            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            print all_predictions
            print lb.classes_


    return jsonify(target=[data_helpers.topicMapToGroup(data['target'])],output=[lb.classes_[int(label)] for label in all_predictions])

if __name__ == '__main__':
    app.run(host='0.0.0.0')