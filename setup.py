#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "matplotlib",
    "scipy",
]

setup_requirements = [
    # TODO(sebastianfrelle): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='dtu_ml_data_mining',
    version='0.1.0',
    description="DTU Course Exercises and codes",
    long_description=readme + '\n\n' + history,
    author="Sebastian Frelle Koch",
    author_email='sebastian.frelle@gmail.com',
    url='https://github.com/sebastianfrelle/dtu_ml_data_mining',
    packages=find_packages(include=['dtu_ml_data_mining']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='dtu_ml_data_mining',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
