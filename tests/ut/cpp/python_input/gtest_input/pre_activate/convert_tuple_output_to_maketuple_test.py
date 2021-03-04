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
tuple_get_item = Primitive(Constants.kTupleGetItem)
LSTM = P.LSTM(input_size=10, hidden_size=2, num_layers=1, has_bias=True, bidirectional=False, dropout=0.0)
add = P.Add()


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_convert_tuple_output_to_maketuple(tag):
    fns = FnDict()

    @fns
    def before(x, h, c, w):
        res = LSTM(x, h, c, w)
        return res

    @fns
    def after(x, h, c, w):
        res = LSTM(x, h, c, w)
        res = make_tuple(
            make_tuple(tuple_get_item(res, 0), tuple_get_item(res, 1), tuple_get_item(res, 2), tuple_get_item(res, 3),
                       tuple_get_item(res, 4)))
        return res

    return fns[tag]
