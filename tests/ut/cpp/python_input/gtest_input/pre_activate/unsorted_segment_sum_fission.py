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
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops import _constants as Constants

make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
unsorted_segment_sum = P.UnsortedSegmentSum()
num_segments = 4
padding = Primitive('Padding')
op_slice = Primitive('Slice')
op_unsorted_segment_sum = Primitive('UnsortedSegmentSum')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_unsorted_segment_sum_fission(tag):
    fns = FnDict()

    @fns
    def before1(input0, input1):
        x = unsorted_segment_sum(input0, input1, num_segments)
        return x

    @fns
    def after1(input0, input1):
        x = padding(input0)
        x = op_unsorted_segment_sum(x, input1)
        x = op_slice(x)
        return make_tuple(x)

    @fns
    def before2(input0, input1):
        x = unsorted_segment_sum(input0, input1, num_segments)
        return x

    @fns
    def after2(input0, input1):
        x = op_unsorted_segment_sum(input0, input1)
        return make_tuple(x)

    return fns[tag]
