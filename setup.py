"""The setup script"""

from setuptools import setup, find_packages

setup(
    name='dtu_ml_data_mining',
    version='0.1.0',
    description="DTU course no. 02450 project source code",
    author="Sebastian Frelle Koch",
    author_email='sebastian.frelle@gmail.com',
    url='https://github.com/sebastianfrelle/dtu_ml_data_mining',
    packages=find_packages(include=['dtu_ml_data_mining']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'xlrd',
    ],
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
)
