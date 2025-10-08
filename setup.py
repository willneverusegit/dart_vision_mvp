"""
Setup configuration for dart_vision_mvp package
"""

from setuptools import setup, find_packages

setup(
    name="dart_vision_mvp",
    version="0.1.0",
    description="CPU-optimized dart vision system",
    author="DoMe",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
)