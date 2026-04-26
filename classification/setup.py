"""
setup.py
--------
Makes the project pip-installable:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="emotisense",
    version="1.0.0",
    author="EmotiSense Research",
    description="Hierarchical Multi-Label Emotion Classification & Crisis Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "nltk>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "app": ["streamlit>=1.24.0", "plotly>=5.14.0"],
        "dev": ["pytest>=7.0", "pytest-cov>=4.0", "black", "isort", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "emotisense-train=src.training:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
