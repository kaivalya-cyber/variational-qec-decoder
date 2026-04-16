"""Setup script for the Variational QEC Decoder package."""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="variational_qec_decoder",
    version="0.1.0",
    author="Variational QEC Team",
    description="Adaptive noise-aware variational quantum error correction decoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/variational-qec/variational_qec_decoder",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pennylane>=0.38",
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "stim>=1.12",
        "pymatching>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=6.0",
            "black>=23.0",
            "mypy>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
