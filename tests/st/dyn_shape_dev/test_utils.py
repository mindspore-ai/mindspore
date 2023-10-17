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

from functools import wraps
from mindspore import nn
import mindspore as ms


ms.set_context(jit_syntax_level=ms.STRICT)


class Net(nn.Cell):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def construct(self, *inputs):
        return self.func(*inputs)


def run_with_cell(fn):
    if fn is None:
        raise ValueError("fn cannot be none!")

    @wraps(fn)
    def wrapper(*args):
        cell_obj = Net(fn)
        return cell_obj(*args)

    return wrapper


def to_cell_obj(fn):
    cell_obj = Net(fn)
    return cell_obj
