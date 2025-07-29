#!/usr/bin/env python
"""
Setup script for PostMath Framework
© 2025 Jesús Manuel Soledad Terrazas. All rights reserved.
"""

from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="postmath",
    version="1.0.0",
    author="Jesús Manuel Soledad Terrazas",
    author_email="jesussoledadt@gmail.com",
    description="PostMath™ - A dual-mode semantic engine that fuses linear NLP with non-linear, cascade-oriented operators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Marte-AI/Postmath-framework",
    packages=find_packages(where=".", include=["postmath*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
        "viz": [
            "matplotlib>=3.5",
            "networkx>=2.8",
            "seaborn>=0.12",
        ],
        "docs": [
            "mkdocs>=1.4",
            "mkdocs-material>=8.5",
            "mkdocstrings[python]>=0.19",
        ],
    },
    entry_points={
        "console_scripts": [
            "postmath=postmath.cli:run_comprehensive_demo",
            "postmath-interactive=postmath.cli:interactive_postmath_demo",
        ],
    },
    include_package_data=True,
    license="PostMath Public Research License v1.0",
    keywords="semantic-analysis nlp ai cascade-dynamics uncertainty-mapping dual-mode-processing",
)