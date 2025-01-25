#!/usr/bin/env python

import os
from setuptools import setup, find_packages


setup(
    name='seakmc_p',
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    version='2.0.0',
    package_data={
        "seakmc_p.input": ["*.yaml"],
    },
    entry_points={
        'console_scripts': ['seakmc_p = seakmc_p.script.seakmc_p:main']
    },
    description='Self Evolution Adaptive Kinetic Monte Carlo',
    author='Tao Liang',
    author_email='xhtliang120@gmail.com',
    url='https://github.com/TaoLiang120/SEAKMC_py_parallel',
)
