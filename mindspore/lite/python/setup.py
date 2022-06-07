#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""setup package."""
import os

from setuptools import setup, find_packages

TOP_DIR = os.getenv('TOP_DIR').replace("\n", "")


def _read_file(filename):
    with open(filename, encoding='UTF-8') as f:
        return f.read()


version = _read_file(TOP_DIR + '/version.txt').replace("\n", "")
readme = _read_file(TOP_DIR + '/mindspore/lite/README.md')

setup(
    name="mindspore_lite",
    version=version,
    author='The MindSpore Authors',
    author_email='contact@mindspore.cn',
    url='https://www.mindspore.cn',
    download_url='https://github.com/mindspore-ai/mindspore/tags',
    project_urls={
        'Sources': 'https://github.com/mindspore-ai/mindspore',
        'Issue Tracker': 'https://github.com/mindspore-ai/mindspore/issues',
    },
    description='MindSpore is a new open source deep learning training/inference '
                'framework that could be used for mobile, edge and cloud scenarios.',
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={'': ['*.py', 'lib/*.so*', '.commit_id', 'include/api/*', 'include/api/callback/*',
                       'include/api/metrics/*', 'include/mindapi/base/*', 'include/registry/converter_context.h',
                       'include/converter.h']},
    include_package_data=True,
    cmdclass={},
    entry_points={},
    python_requires='>=3.7',
    install_requires=['numpy >= 1.17.0'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='mindspore lite',
)
