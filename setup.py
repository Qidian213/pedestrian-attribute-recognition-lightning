#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='par',
    version='0.0.1',
    description='Pedestrian attribute recognition.',
    author='Xing Zhao LEE',
    url='https://github.com/xingzhaolee/pedestrian-attribute-recognition-lightning',
    install_requires=[
            'pytorch-lightning'
    ],
    packages=find_packages()
)
