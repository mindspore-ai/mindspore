#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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
"""setup_package."""
import os
import stat
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py

package_name = 'mindspore'
version = '0.1.0'
author = 'The MindSpore Authors'
author_email = 'contact@mindspore.cn'
home_page = 'https://www.mindspore.cn'

backend_policy = os.getenv('BACKEND_POLICY')
commit_id = os.getenv('COMMIT_ID').replace("\n", "")

pwd = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(pwd, 'build/package')

def write_version(file):
    file.write("__version__ = '{}'\n".format(version))

def write_config(file):
    file.write("__backend__ = '{}'\n".format(backend_policy))

def write_commit_file(file):
    file.write("__commit_id__ = '{}'\n".format(commit_id))

def build_depends():
    """generate python file"""
    version_file = os.path.join(pwd, 'build/package/mindspore', 'version.py')
    with open(version_file, 'w') as f:
        write_version(f)

    version_file = os.path.join(pwd, 'mindspore/', 'version.py')
    with open(version_file, 'w') as f:
        write_version(f)

    config_file = os.path.join(pwd, 'build/package/mindspore', 'default_config.py')
    with open(config_file, 'w') as f:
        write_config(f)

    config_file = os.path.join(pwd, 'mindspore/', 'default_config.py')
    with open(config_file, 'w') as f:
        write_config(f)

    commit_file = os.path.join(pwd, 'build/package/mindspore', '.commit_id')
    with open(commit_file, 'w') as f:
        write_commit_file(f)

    commit_file = os.path.join(pwd, 'mindspore/', '.commit_id')
    with open(commit_file, 'w') as f:
        write_commit_file(f)

descriptions = 'An AI computing framework that supports development for AI applications in all scenarios.'

requires = [
        'numpy >= 1.17.0',
        'protobuf >= 3.8.0',
        'asttokens >= 1.1.13',
        'pillow >= 6.2.0',
        'scipy == 1.3.3',
        'easydict >= 1.9',
        'sympy >= 1.4',
        'cffi >= 1.13.2',
        'decorator >= 4.4.0'
    ],

package_datas = {
    '': [
        '*.so*',
        'lib/*.so*',
        'lib/*.a',
        '.commit_id',
    ]
}

build_depends()

def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)

class EggInfo(egg_info):
    """Egg info."""
    def run(self):
        super().run()
        egg_info_dir = os.path.join(pkg_dir, 'mindspore.egg-info')
        update_permissions(egg_info_dir)

class BuildPy(build_py):
    """BuildPy."""
    def run(self):
        super().run()
        mindspore_dir = os.path.join(pkg_dir, 'build', 'lib', 'mindspore')
        update_permissions(mindspore_dir)
        mindspore_dir = os.path.join(pkg_dir, 'build', 'lib', 'akg')
        update_permissions(mindspore_dir)

setup(
    python_requires='>=3.7',
    name=package_name,
    version=version,
    author=author,
    author_email=author_email,
    url=home_page,
    packages=find_packages(),
    package_data=package_datas,
    include_package_data=True,
    cmdclass={
        'egg_info': EggInfo,
        'build_py': BuildPy,
    },
    install_requires=requires,
    description=descriptions,
    license='Apache 2.0',
)
