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
from mindspore.ops.operations import _grad_ops as G

stridedslicegrad = G.StridedSliceGrad()
backend_stridedslicegrad = Primitive('StridedSliceGrad')
make_tuple = Primitive('MakeTuple')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_const_to_attr_strided_slice_grad(tag):
    fns = FnDict()

    @fns
    def before(x):
        return stridedslicegrad(x, (16, 128, 1024), (0, 0, 0), (16, 1, 1024), (1, 1, 1))

    @fns
    def after(x):
        res = backend_stridedslicegrad(x)
        return make_tuple(res)

    return fns[tag]
