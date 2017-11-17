#!/bin/sh

## state-of-the-art-result-for-machine-learning-problems
git subtree add --prefix .resources/reports/state-of-the-art-result-for-machine-learning-problems \
	https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems \
	--squash master

## Kaggle - multi-class-text-classification-cnn
git subtree add --prefix .resources/kaggle/consumer-finance/jiegzhan/multi-class-text-classification-cnn \
	https://github.com/jiegzhan/multi-class-text-classification-cnn \
	--squash master

## TensorFlow - image-classification-rnn
git subtree add --prefix .resources/deeplearning/images/jiegzhan/image-classification-rnn \
	https://github.com/jiegzhan/image-classification-rnn \
	--squash master

## Tensorflow - document classification
git subtree add --prefix .resources/deeplearning/text/classification/tensorflow-font2char2word2sent2doc \
	https://github.com/raviqqe/tensorflow-font2char2word2sent2doc \
	--squash master

## Tensorflow - Entity Recognition
git subtree add --prefix .resources/deeplearning/text/entity-recognition/tensorflow-entrec \
	https://github.com/raviqqe/tensorflow-entrec \
	--squash master

## Tensorflow - Deep Reinforcement Learning
git subtree add --prefix .resources/deeplearning/learn-reinforcement/deep-rl-tensorflow \
	https://github.com/carpedm20/deep-rl-tensorflow \
	--squash master