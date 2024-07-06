from setuptools import setup, find_packages

setup(
    name='transformers_tf',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'tensorflow-datasets',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'train_transformer=scripts.train_transformer:main',
        ],
    },
)
