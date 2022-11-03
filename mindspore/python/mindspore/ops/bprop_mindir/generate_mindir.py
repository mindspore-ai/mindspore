# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Generate the mindir for bprop"""
from __future__ import absolute_import
import os
import shutil
import argparse
import logging

import mindspore.ops._grad as g
from mindspore._c_expression import _export_bprop_mindir
from mindspore.ops._grad.grad_base import bprop_getters, bprops

logging.getLogger().setLevel(logging.INFO)
os.environ['MS_DEV_EXPORT_BPROP_MINDIR'] = '1'


def run_generate(bprop_mindir_install_dir, bprop_map, force_update):
    for op_name in bprop_map.keys():
        if not isinstance(op_name, str):
            continue
        if os.path.isfile(os.path.join(bprop_mindir_install_dir, op_name + "_bprop.mindir")):
            _export_bprop_mindir(op_name, force_update)


def run_generate_with_op_name(op_name, force_update):
    _export_bprop_mindir(op_name, force_update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bprop mindir generator")
    parser.add_argument('--mindspore_path', type=str, default=None,
                        help="The absolute path of the mindspore root directory where the bprop source files has been \
                        modified. If not specified, it will find the bprop source files in your mindspore installed \
                        path. Default: None.")
    parser.add_argument('--op', type=str, default=None,
                        help="The name of the operator whose bprop is to be transformed to mindir file. If not \
                        specified, it will generate all the mindir files of bprop which has already been transform \
                        to mindir. Default: None.")
    parser.add_argument('--force', type=bool, default=False,
                        help="Whether to force to generate mindir file of a bprop function regardless of nothing \
                        changed. Default: False.")

    args_opt = parser.parse_args()
    # mindspore/ops/_grad/__init__.py
    BPROP_PATH = g.__file__
    bprop_installed_dir = BPROP_PATH[: BPROP_PATH.rindex('/')]
    bprop_mindir_export_dir = os.path.join(bprop_installed_dir, "..", "bprop_mindir")

    mindspore_path = args_opt.mindspore_path
    bprop_src_dir = None
    bprop_mindir_src_dir = None
    if mindspore_path is not None:
        mindspore_path = mindspore_path.rstrip('/')
        python_ops_dir = os.path.join(mindspore_path, "mindspore", "python", "mindspore", "ops")
        bprop_src_dir = os.path.join(python_ops_dir, "_grad")
        bprop_mindir_src_dir = os.path.join(python_ops_dir, "bprop_mindir")

    copy_flag = bprop_src_dir is not None and bprop_src_dir != bprop_installed_dir
    # If the specified bprop source directory is not on the mindspore installed path,
    # copy the bprop source files to the installed path.
    BACKUP_SUFFIX = "_generate_bak"
    if copy_flag:
        shutil.rmtree(bprop_installed_dir + BACKUP_SUFFIX, ignore_errors=True)
        os.rename(bprop_installed_dir, bprop_installed_dir + BACKUP_SUFFIX)
        os.mkdir(bprop_installed_dir)
        ls = os.listdir(bprop_src_dir)
        for line in ls:
            file_path = os.path.join(bprop_src_dir, line)
            if os.path.isfile(file_path):
                shutil.copy(file_path, bprop_installed_dir)
                logging.info("copied: %s", file_path)

    force = args_opt.force
    op = args_opt.op
    if op is None:
        run_generate(bprop_mindir_export_dir, bprop_getters, force)
        run_generate(bprop_mindir_export_dir, bprops, force)
    else:
        run_generate_with_op_name(op, force)

    # If the specified bprop source directory is not on the mindspore installed path,
    # copy the generated mindir files to the mindir directory relative to the specified path.
    if copy_flag:
        shutil.rmtree(bprop_installed_dir)
        os.rename(bprop_installed_dir + BACKUP_SUFFIX, bprop_installed_dir)
        ls = os.listdir(bprop_mindir_export_dir)
        for line in ls:
            file_path = os.path.join(bprop_mindir_export_dir, line)
            if file_path.endswith(".mindir") and os.path.isfile(file_path):
                os.chmod(file_path, 0o664)
                shutil.copy(file_path, bprop_mindir_src_dir)
                logging.info("copied: %s", file_path)

        logging.info("The new bprop mindir files has been generated in the path \"%s\"", bprop_mindir_src_dir)
    else:
        logging.info("The new bprop mindir files has been generated in the path \"%s\", "
                     "copy the *.mindir to your mindspore path or PYTHONPATH if necessary.", bprop_mindir_export_dir)

    del os.environ['MS_DEV_EXPORT_BPROP_MINDIR']
