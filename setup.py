#!/usr/bin/env python3
"""
Setup script for PlayNexus Satellite Toolkit
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="playnexus-satellite-toolkit",
    version="1.1.0",
    author="PlayNexus",
    author_email="playnexushq@gmail.com",
    description="A comprehensive, production-ready toolkit for satellite imagery analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/playnexus/satellite-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "playnexus-satellite=playnexus_satellite_toolkit:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
        "assets": ["*"],
        "docs": ["*"],
        "gee": ["*.js"],
        "viewer": ["*.html", "*.css", "*.js"],
    },
    zip_safe=False,
)
