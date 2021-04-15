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
"""setup package."""
import os
import stat
import platform

from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py

version = '1.2.0'

backend_policy = os.getenv('BACKEND_POLICY')
device_target = os.getenv('BACKEND_TARGET')
commit_id = os.getenv('COMMIT_ID').replace("\n", "")
package_name = os.getenv('MS_PACKAGE_NAME').replace("\n", "")

pwd = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(pwd, 'build/package')


def _read_file(filename):
    with open(os.path.join(pwd, filename), encoding='UTF-8') as f:
        return f.read()


readme = _read_file('README.md')


def _write_version(file):
    file.write("__version__ = '{}'\n".format(version))


def _write_config(file):
    file.write("__backend__ = '{}'\n".format(backend_policy))


def _write_commit_file(file):
    file.write("__commit_id__ = '{}'\n".format(commit_id))


def _write_package_name(file):
    file.write("__package_name__ = '{}'\n".format(package_name))


def _write_device_target(file):
    file.write("__device_target__ = '{}'\n".format(device_target))


def build_dependencies():
    """generate python file"""
    version_file = os.path.join(pkg_dir, 'mindspore', 'version.py')
    with open(version_file, 'w') as f:
        _write_version(f)

    version_file = os.path.join(pwd, 'mindspore', 'version.py')
    with open(version_file, 'w') as f:
        _write_version(f)

    config_file = os.path.join(pkg_dir, 'mindspore', 'default_config.py')
    with open(config_file, 'w') as f:
        _write_config(f)

    config_file = os.path.join(pwd, 'mindspore', 'default_config.py')
    with open(config_file, 'w') as f:
        _write_config(f)

    target = os.path.join(pkg_dir, 'mindspore', 'default_config.py')
    with open(target, 'a') as f:
        _write_device_target(f)

    target = os.path.join(pwd, 'mindspore', 'default_config.py')
    with open(target, 'a') as f:
        _write_device_target(f)

    package_info = os.path.join(pkg_dir, 'mindspore', 'default_config.py')
    with open(package_info, 'a') as f:
        _write_package_name(f)

    package_info = os.path.join(pwd, 'mindspore', 'default_config.py')
    with open(package_info, 'a') as f:
        _write_package_name(f)

    commit_file = os.path.join(pkg_dir, 'mindspore', '.commit_id')
    with open(commit_file, 'w') as f:
        _write_commit_file(f)

    commit_file = os.path.join(pwd, 'mindspore', '.commit_id')
    with open(commit_file, 'w') as f:
        _write_commit_file(f)


build_dependencies()

required_package = [
    'numpy >= 1.17.0',
    'protobuf >= 3.8.0',
    'asttokens >= 1.1.13',
    'pillow >= 6.2.0',
    'scipy >= 1.5.2',
    'easydict >= 1.9',
    'sympy >= 1.4',
    'cffi >= 1.12.3',
    'wheel >= 0.32.0',
    'decorator >= 4.4.0',
    'setuptools >= 40.8.0',
    'astunparse >= 1.6.3',
    'packaging >= 20.0',
    'psutil >= 5.6.1'
]

package_data = {
    '': [
        '*.so*',
        '*.pyd',
        '*.dll',
        'bin/*',
        'lib/*.so*',
        'lib/*.a',
        'lib/*.dylib*',
        '.commit_id',
        'config/*',
        'include/*',
        'include/*/*',
        'include/*/*/*',
        'include/*/*/*/*'
    ]
}


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    if platform.system() == "Windows":
        return

    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE |
                     stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
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
        mindspore_dir = os.path.join(pkg_dir, 'build', 'lib', 'mindspore', '_akg')
        update_permissions(mindspore_dir)


setup(
    name=package_name,
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
    package_data=package_data,
    include_package_data=True,
    cmdclass={
        'egg_info': EggInfo,
        'build_py': BuildPy,
    },
    entry_points={
        'console_scripts': [
            'cache_admin=mindspore.dataset.engine.cache_admin:main',
        ],
    },
    python_requires='>=3.7',
    install_requires=required_package,
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
    keywords='mindspore machine learning',
)
