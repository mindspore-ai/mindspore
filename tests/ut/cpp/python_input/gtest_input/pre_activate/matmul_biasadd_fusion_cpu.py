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
from mindspore.ops import _constants as Constants
from mindspore.ops import operations as P

matmul = P.MatMul()
biasadd = P.BiasAdd()
prim_after_fusion = Primitive('FusedMatMulBiasAdd')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fn_dict.get(name)


def test_matmul_biasadd_fusion_cpu(tag):
    """
    Feature: Test MatMulBiasAddFusionCpu pass
    Description: Test MatMulBiasAddFusionCpu pass
    Expectation: The graph after fusion is as expected when it meets the pattern of the pass.
    """
    fns = FnDict()

    @fns
    def before(input0, input1, input2):
        res = matmul(input0, input1)
        res = biasadd(res, input2)
        return res

    @fns
    def after(input0, input1, input2):
        res = prim_after_fusion(input0, input1, input2)
        return make_tuple(res)

    return fns[tag]
