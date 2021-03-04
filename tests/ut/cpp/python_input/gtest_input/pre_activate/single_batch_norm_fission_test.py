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
BatchNorm = P.BatchNorm(is_training=True)
BNTrainingReduce = Primitive('BNTrainingReduce')
BNTrainingUpdateV3 = Primitive('BNTrainingUpdateV3')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_single_batch_norm_fission(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4):
        batch_norm = BatchNorm(input0, input1, input2, input3, input4)
        item0 = tuple_getitem(batch_norm, 0)
        item1 = tuple_getitem(batch_norm, 1)
        item2 = tuple_getitem(batch_norm, 2)
        item3 = tuple_getitem(batch_norm, 3)
        item4 = tuple_getitem(batch_norm, 4)
        output = make_tuple(item0, item1, item2, item3, item4)
        return output

    @fns
    def after(input0, input1, input2, input3, input4):
        reduce = BNTrainingReduce(input0)
        reduce_item0 = tuple_getitem(reduce, 0)
        reduce_item1 = tuple_getitem(reduce, 1)
        update = BNTrainingUpdateV3(input0, reduce_item0, reduce_item1, input1, input2)
        update_item0 = tuple_getitem(update, 0)
        update_item1 = tuple_getitem(update, 1)
        update_item2 = tuple_getitem(update, 2)
        update_item3 = tuple_getitem(update, 3)
        update_item4 = tuple_getitem(update, 4)
        output = make_tuple(update_item0, update_item1, update_item2, update_item3, update_item4)
        return make_tuple(output)

    return fns[tag]
