"""
build mslite_bench whl
"""
from setuptools import setup, find_packages
with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mslite_bench',
    version='0.0.1-alpha',
    description='performance and accuracy tools for multiple framework model infer',
    long_description=long_description,
    url='mslite_bench url',
    packages=find_packages(),
    py_modules=['mslite_bench'],
    keywords='mslite_bench',
    install_requires=required,
    python_requires='>=3.7'
)
