# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="propepdesigner",
    version="1.0",
    author="Hongyan Yin",
    author_email="18810910113@163.com",
    description="ProPepDesigner: AI driving  de novo design of long-acting GIPR/GLP-1R/GCGR triple agonists for obesity therapy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiaodaoyhy/ProPepDesigner",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires="==3.8.13"
)