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
"""Arguments for configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


def str2bool(v):
    """
    str2bool
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup():
    """ArgumentGroup"""

    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, dtype, default, helpinfo,
                positional_arg=False, **kwargs):
        """add_arg"""
        prefix = "" if positional_arg else "--"
        dtype = str2bool if dtype == bool else dtype
        self._group.add_argument(
            prefix + name,
            default=default,
            type=dtype,
            help=helpinfo + ' Default: %(default)s.',
            **kwargs)

def build_common_arguments():
    """build_common_arguments"""
    parser = argparse.ArgumentParser(__doc__)
    model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
    model_g.add_arg("job", str, 'SST-2', "job to be trained (SST-2 or Sem-L)")
    model_g.add_arg("data_url", str, "./data/", "data url")
    model_g.add_arg("train_url", str, "./data/", "output url")
    model_g.add_arg("ckpt", str, "roberta.ckpt", "ckpt url")
    model_g.add_arg(
        "is_modelarts_work", str, "false", "Whether modelarts online work.")
    model_g.add_arg("batch_size", int, 24, "batch_size")

    args = parser.parse_args()
    return args


def replace_none(params):
    """replace_none"""
    if params == "None":
        return None
    if isinstance(params, dict):
        for key, value in params.items():
            params[key] = replace_none(value)
            if key == "split_char" and isinstance(value, str):
                try:
                    value = chr(int(value, base=16))
                    print("ord(value): ", ord(value))
                except IOError:
                    pass
                params[key] = value
        return params
    if isinstance(params, list):
        return [replace_none(value) for value in params]
    return params
