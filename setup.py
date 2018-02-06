"""The setup script"""

from setuptools import setup, find_packages

setup(
    name='dtu_ml_data_mining',
    description="DTU course no. 02450 project source code",
    author='Sebastian Frelle Koch',
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
)
