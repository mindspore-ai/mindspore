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
from mindspore.ops import operations as P

Dropout = P.Dropout(0.5, 1, 2)
ReLU = P.ReLU()
Add = P.Add()


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        try:
            result = self.fn_dict[name]
        except KeyError:
            result = None
        return result


def add_dropout_usage_attrs_graph(tag):
    fns = FnDict()

    @fns
    def only_first_output(input0):
        first_output, _ = Dropout(input0)
        return ReLU(first_output)

    @fns
    def only_second_output(input0):
        _, second_output = Dropout(input0)
        return ReLU(second_output)

    @fns
    def all_output(input0):
        first_output, second_output = Dropout(input0)
        return Add(ReLU(first_output), ReLU(second_output))

    return fns[tag]
