#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import re
import MySQLdb
from sklearn import preprocessing
from bs4 import BeautifulSoup
from langdetect import detect
import spacy
import en_core_web_sm
import sys
import pickle
import os


nlp = en_core_web_sm.load()

def replace_entity(span,replacement):
    i = 1
    for token in span:
        if i == span.__len__():
            token.lemma_ = replacement
        else:
            token.lemma_ = u''
        i += 1

def customize_rule(doc):
    for ent in doc.ents:
        if ent.label_ == u'PERSON':
            replace_entity(ent,u'PERSON')
        if ent.label_ == u'DATE':
            replace_entity(ent,u'DATE')
        if ent.label_ == u'TIME':
            replace_entity(ent,u'TIME')

    for token in doc:
        if token.like_url:
            token.lemma_ = u'URL'
        if token.is_digit and token.lemma_ not in [u'-DATE-',u'-TIME-',u'']:
            token.lemma_ = u'NUM'
        if token.lemma_ == u'PRON':
            token.lemma_ = token.text

nlp.pipeline = [nlp.tagger,nlp.entity,customize_rule]

def punct_space(token):
    return token.is_space

def preparation(corpus):
    return [token.lemma_ for token in nlp(BeautifulSoup(corpus,"html.parser").get_text()) if not punct_space(token)]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9;.,!?’'`:/¥$€@\s]", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = u' '.join(preparation(string.decode('utf8', 'ignore')))
    return string


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]

def load_data_and_labels_from_db():
    db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
    cursor = db.cursor()
    dict={}
    x_text =[]
    y=[]
    x_id =[]
    previous_y=[]
    #sql = "SELECT t1_final,t2_final,subject,body FROM text_source_data WHERE site in ('EBAY_AU','EBAY_MAIN','EBAY_CA','EBAY_UK') " \
          #"and channel='Email' and body !='eBP Automation Request'"
    sql = "SELECT DISTINCT(sr_number),t1_final,t2_final ,subject,body FROM nice_text_source_data WHERE t2_final in ('eBay Account Information - CCR', \
          'Buyer Protect High ASP Claim','VeRO - CCR','Logistics - CCR','Buyer Protection Escalate SNAD','High Risk','Buyer Protection Appeal SNAD','Seller Risk Management', \
          'Buyer Protection Appeal INR','Returns','Buyer Protection Programs Qs','Buyer Protection Escalate INR','Selling Limits - CCR','Buyer Protection Case Qs', \
          'Contact Trading Partner - CCR','Paying for Items','Defect Appeal','Specialty Selling Approvals')"

    try:
        cursor.execute(sql)
        results = cursor.fetchall()

    except:
        sys.stdout.write("Error: unable to fecth data"+ '\n')

    db.close()
    i=0
    sys.stdout.write("Value is %s" % i)
    sys.stdout.write('\n')
    for row in results:
        text = (row[3] + ' ' + row[4]).decode('utf8', 'ignore')
        try:
            if text!='' and detect(text)=='en':
                x_text.append(clean_str(text))
                x_id.append(row[0])
                i=i+1
                sys.stdout.write("Value is %s" % i)
                sys.stdout.write('\n')
                previous_y.append(row[2])
        except:
            sys.stdout.write(row[0])

    lb = preprocessing.LabelBinarizer()
    y=lb.fit_transform(list(previous_y))

    f = open('y_target.pickle', 'wb')
    pickle.dump(lb, f)
    f.close()

    sys.stdout.write(lb.inverse_transform(y))
    return [x_text, y,np.array(x_id)]


def topicMapToGroup(topic):
    if topic in ['Bidding/Buying Items','Account Safety - CCR','Unknown','Buyer Loyalty Programs','eBay Account Information - CCR','Listing Ended/Removed - Buyer','eBay Partner Sites - CCR',
        'Paying for Items','Forgot User ID or  Password',
        'Search - Buying','Payment Service Account Setup',
        'Seller Suspended - Buyer',	'Registering an Account',
	    'Site Features - CCR']:
        return 'US Buying and General'
    elif topic in ['Advanced Applications']:
        return 'US Advanced Apps'
    elif topic in ['Account Closure - CCR',	'Billing Account on Hold',
        'Business Development - CCR',	'Billing Invoice',
        'Completing a Sale - CCR',	'Billing Refunds - CCR',
        'Listing Queries - CCR',	'Collections - CCR',
        'Managing Bidders/Buyers - CCR',	'eBay Fees - CCR',
        'Marketing Promotions - CCR',	'Non-Payment Suspension - CCR',
        'Search - Selling'	,'Paying eBay',
        'Selling Tools - CCR',	'Payment Service Account Funds',
        'Shipping - CCR'	,'Payment Service Fees',
        'Stores/Shops - CCR',	'Request a Credit']:
        return 'US Selling'
    elif topic in ['Buyer Protection Case Qs',
        'Buyer Protection Program Qs',
        'Buyer Protection Refunds',
        'Cancel Transaction',
        'Contact Trading Partner - CCR',
        'Payment Service Dispute',
        'Seller Protection Policy',
        'Unpaid Item - Seller']:
        return 'US M2M Mediation'
    elif topic in ['Buyer Protection Escalate INR',
        'Buyer Protection Escalate SNAD',
        'Returns',
        'UPI Appeal - CCR']:
        return 'US M2M Escalation'
    elif topic in ['Buyer Protection Appeal INR',
        'Buyer Protection Appeal SNAD',
        'Defect Appeals',
        'Defect Basic Process']:
        return 'US M2M Appeals'
    elif topic in ['Buyer Protect High ASP Claim']:
        return 'US M2M High ASP Claim'
    elif topic in ['Account Restriction',
        'Account Suspension',
        'Buying - Rules & Policies',
        'Funds Availability - CCR',
        'High Risk',
        'INV Policies',
        'Known Good',
        'Law Enforcement - CCR',
        'Off Site Transaction - CCR',
        'Report a Member/Listing',
        'Spoof Email']:
        return 'US e2M Account'
    elif topic in ['Account Takeover']:
        return 'ATO Global'
    elif topic in ['Buying Limits - CCR'
        'Seller Vetting Restriction',
        'Selling Limits - CCR',
        'Selling Performance']:
        return 'US e2M Limits'
    elif topic in ['CIT - Counterfeit',
        'Infringement - CCR',
        'List Practices',
        'Listing Removed - CCR',
        'Prohibited & Restricted Item']:
        return 'US e2M Listing'
    elif topic in ['VeRO - CCR']:
        return 'US e2M VeRO'
    else:
        return topic

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    return datasets



def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors
