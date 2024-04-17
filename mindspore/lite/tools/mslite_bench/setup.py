# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
build mslite_bench whl
"""
from setuptools import setup, find_packages
with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='mslite_bench',
    version='0.0.1-alpha',
    description='performance and accuracy tools for multiple framework model infer',
    long_description='Debug and optimizer tool for mindspore lite',
    url='mslite_bench url',
    packages=find_packages(),
    py_modules=['mslite_bench'],
    keywords='mslite_bench',
    install_requires=required,
    python_requires='>=3.7'
)
