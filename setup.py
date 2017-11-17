#!/usr/bin/env python3

from setuptools import setup

with open('requirements.txt', 'r') as f:
    reqs = f.read()

setup(
    name="multi-class-text-classification-cnn-rnn",
    version="0.0.1",
    description="Classify Kaggle San Francisco Crime Description into 39 classes.",
    author="Jie Zhang",
    author_email="jiegzhan@gmail.com",
    url="https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn",
    install_requires=reqs,
    packages=["tf-multiclass-text"],
    entry_points={
        'console_scripts': [
            "predict = scripts.predict:main",
            "train = scripts.train:main",
            "server = scripts.server:main",
        ]
    },
    package_data={
        '': ['*.json','*.tar.gz','*.zip','*.gz']
    }
)