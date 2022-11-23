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
from mindspore.ops import Primitive
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import _constants as Constants

MakeTuple = Primitive('MakeTuple')
GetTupleItem = Primitive(Constants.kTupleGetItem)
FlattenConcat = inner.FlattenConcat()
Flatten = Primitive('Flatten')
Concat = Primitive('Concat')
ConcatD = Primitive('ConcatD')


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


def flatten_concat_fission_graph(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4):
        outputs = FlattenConcat(MakeTuple(input0, input1, input2, input3, input4))
        return outputs

    @fns
    def after(input0, input1, input2, input3, input4):
        flatten0 = Flatten(input0)
        flatten1 = Flatten(input1)
        flatten2 = Flatten(input2)
        flatten3 = Flatten(input3)
        flatten4 = Flatten(input4)
        concat0 = ConcatD(flatten0, flatten2, flatten4)
        concat1 = ConcatD(flatten1, flatten3)
        outputs = MakeTuple(concat0, concat1)
        multi_output_tuple = MakeTuple(GetTupleItem(outputs, 0), GetTupleItem(outputs, 1))
        final_tuple = MakeTuple(multi_output_tuple)
        return final_tuple

    return fns[tag]
