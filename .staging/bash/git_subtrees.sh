#!/bin/sh

## state-of-the-art-result-for-machine-learning-problems
git subtree add --prefix .resources/reports/state-of-the-art-result-for-machine-learning-problems \
	https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems \
	--squash master

## Kaggle - multi-class-text-classification-cnn
git subtree add --prefix .resources/kaggle/consumer-finance/jiegzhan/multi-class-text-classification-cnn \
	https://github.com/jiegzhan/multi-class-text-classification-cnn \
	--squash master

## Tensorflow - image-classification-rnn
git subtree add --prefix .resources/deeplearning/images/jiegzhan/image-classification-rnn \
	https://github.com/jiegzhan/image-classification-rnn \
	--squash master