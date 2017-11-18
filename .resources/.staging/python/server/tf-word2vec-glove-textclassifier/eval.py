#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

from langdetect import detect

import pickle
import logging

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
#tf.flags.DEFINE_string("checkpoint_dir", "/Users/zqiao/PycharmProjects/cnn-text-classification-tf/runs/1492410542/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1498107052/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

f = open('./y_target.pickle', 'rb')
lb = pickle.load(f)

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    file = open('./dev_data.pickle', 'rb')
    x_test,y_test,x_id_test= pickle.load(file)
    y_test = np.argmax(y_test,axis=1)
else:
    x_raw=[]
    y_test=[]
    ids=[]
    y_raw=[]
    with open("./eval.csv", 'rU') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['subject'] is None:
                row['subject'] = ''
            if row['body'] is None:
                row['body'] = ''
            text = (row['subject']+' '+row['body']).decode('utf8','ignore')
            try:
                if row['body']!='' and row['body']!='eBP Automation Request' and row['body']!='NULL' and detect(text)=='en':
                    ids.append(row['ID'])
                    x_raw.append(data_helpers.clean_str(text))
                    y_raw.append(data_helpers.topicMapToGroup(row['topic2']))
            except:
                print row['ID']

    print y_raw
    print lb.transform(y_raw)
    y_test = np.argmax(lb.transform(y_raw),axis=1)
    print y_test
    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
print FLAGS.checkpoint_dir
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

#checkpoint_file = FLAGS.checkpoint_dir+'model-3800'
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
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

        print y_test
        print all_predictions
        print lb.classes_
        #print x_id_test

        print tf.contrib.metrics.confusion_matrix(y_test, all_predictions, len(lb.classes_)).eval()

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))

    print("Total number of predicted examples: {}".format(len(all_predictions)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))



# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(ids),np.array(x_raw), np.array(y_raw),[lb.classes_[int(label)] for label in all_predictions]))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)