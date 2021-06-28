# Copyright 2021 Huawei Technologies Co., Ltd
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
import os
import shutil
import argparse
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.ops._grad as g

context.set_context(mode=context.GRAPH_MODE)
os.environ['GENERATE_MINDIR'] = '1'


class NetRelu(nn.Cell):
    def __init__(self):
        super(NetRelu, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


class NetIdentity(nn.Cell):
    def __init__(self):
        super(NetIdentity, self).__init__()
        self.identity = P.Identity()

    def construct(self, x):
        return self.identity(x)


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout


def test_relu():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    relu = NetRelu()
    grad = GradNet(relu)
    grad(x)


def test_identity():
    x = Tensor(np.array([1, 2, 3, 4]).astype(np.int64))
    identity = NetIdentity()
    grad = GradNet(identity)
    grad(x)


def run_generate():
    test_relu()
    test_identity()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bprop generator")
    parser.add_argument('--mindspore_path', type=str, default=None,
                        help="The absolute path of the mindspore root directory where the bprop source files has been \
                        modified. If not specified, it will find the bprop source files in your mindspore installed \
                        path. Default: None.")

    args_opt = parser.parse_args()
    # mindspore/ops/_grad/__init__.py
    bprop_path = g.__file__
    bprop_installed_dir = bprop_path[: bprop_path.rindex('/')]
    bprop_mindir_export_dir = bprop_installed_dir + "/../bprop_mindir"

    mindspore_path = args_opt.mindspore_path
    bprop_src_dir = None
    bprop_mindir_src_dir = None
    if not mindspore_path is None:
        mindspore_path = mindspore_path.rstrip('/')
        bprop_src_dir = mindspore_path + "/mindspore/ops/_grad"
        bprop_mindir_src_dir = mindspore_path + "/mindspore/ops/bprop_mindir"

    copy_flag = not bprop_src_dir is None and bprop_src_dir != bprop_installed_dir
    # If the specified bprop source directory is not on the mindspore installed path,
    # copy the bprop source files to the installed path.
    backup_suffix = "_generate_bak"
    if copy_flag is True:
        shutil.rmtree(bprop_installed_dir + backup_suffix, ignore_errors=True)
        os.rename(bprop_installed_dir, bprop_installed_dir + backup_suffix)
        os.mkdir(bprop_installed_dir)
        ls = os.listdir(bprop_src_dir)
        for line in ls:
            file_path = os.path.join(bprop_src_dir, line)
            if os.path.isfile(file_path):
                print("copy: " + file_path)
                shutil.copy(file_path, bprop_installed_dir)

    run_generate()

    # If the specified bprop source directory is not on the mindspore installed path,
    # copy the generated mindir files to the mindir directory relative to the specified path.
    if copy_flag is True:
        shutil.rmtree(bprop_installed_dir)
        os.rename(bprop_installed_dir + backup_suffix, bprop_installed_dir)
        ls = os.listdir(bprop_mindir_export_dir)
        for line in ls:
            file_path = os.path.join(bprop_mindir_export_dir, line)
            if file_path.endswith(".mindir") and os.path.isfile(file_path):
                print("copy: " + file_path)
                shutil.copy(file_path, bprop_mindir_src_dir)

        print("The new bprop mindir files has been generated in the path \"" + bprop_mindir_src_dir +
              "\".")
    else:
        print("The new bprop mindir files has been generated in the path \"" + bprop_mindir_export_dir +
              "\", copy the *.mindir to your PYTHONPATH if necessary.")
