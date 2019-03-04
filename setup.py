#!/usr/bin/env python3
# flake8: noqa
from os.path import dirname, realpath
from setuptools import find_packages, setup, Distribution
import setuptools.command.build_ext as _build_ext
import subprocess

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name='lqr',
    version='0.0.1',
    distclass=BinaryDistribution,
    packages=find_packages(),
    description=("A fast sampler for Least Squares"),
    long_description=open("README.md").read(),
    zip_safe=False,
)
