#!/usr/bin/env python

import os
from setuptools import setup, find_packages
#module_dir = os.path.dirname(os.path.abspath(__file__))
##with open('README.rst') as f:
##    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name = 'seakmc_p',
    packages = find_packages(exclude=('tests','docs')),
    include_package_data = True,
    version = '2.0.0',
##  install_requires = ['anaconda>=3.0', 'pycrypto>=1.0', 'monty>=0.7.2',
##                      'matplotlib>=1.4.2', 'websocket_client>=0.1', 'nose>=1.3',
##                      'scipy==0.14.0', 'pandas_market_calendars>=0.1'],
##  extras_require = {'doc': ['codecov>=2.0', 'sphinx>=1.3.1']},
    package_data={
        "seakmc_p.input": ["*.yaml"],
    },
    entry_points={
        'console_scripts': ['seakmc_p = seakmc_p.script.seakmc_p:main']
        },
    license = license,
    description = 'Self Evolution Adaptive Kinetic Monte Carlo',
    author = 'Tao Liang',
    author_email = 'xhtliang120@gmail.com',
    url = 'https://github.com/TaoLiang120/SEAKMC_py_parallel',
##  download_url = 'https://github.com/TaoLiang120/SEAKMC_py_parallel.git',
)
