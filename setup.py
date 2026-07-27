"""
Setup script for installing the deep_learning_pipeline package.

"""

from setuptools import setup, find_packages


setup(
    name="deep_learning_pipeline",
    version="0.1.0",
    description="A structured deep learning pipeline for coursework and projects",
    author="mahdi mansouri",
    author_email="mmhdimansouri@gmail.com",
    url="https://github.com/mehtimans/deep_learning_pipeline",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "matplotlib",
        "numpy>=1.20",
        "tensorboard",
        # PyTorch must be installed separately (see https://pytorch.org)
    ],
)