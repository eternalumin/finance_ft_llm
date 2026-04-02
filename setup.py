"""
Setup script for Earnings Call Intelligence System
=================================================

This package provides a fine-tuned LLM for earnings call analysis,
including beat/miss prediction, financial Q&A, and metric extraction.

Usage:
    pip install -e .
    
Or install directly:
    pip install -r requirements.txt
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="earnings-intelligence",
    version="1.0.0",
    author="Earnings Intelligence Team",
    author_email="example@email.com",
    description="Fine-tuned LLM for Earnings Call Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eternalumin/finance_ft_llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "datasets>=2.15.0",
        "pandas>=2.0.0",
        "gradio>=4.0.0",
        "huggingface-hub>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
