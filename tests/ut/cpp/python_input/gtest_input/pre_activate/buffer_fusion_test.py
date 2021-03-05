# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.common.dtype as mstype
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G

Conv = P.Conv2D(out_channel=64, kernel_size=7, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1)
Relu = P.ReLU()
Fusion = Primitive('FusionOp')
Reduce = P.ReduceOp()
Biasadd = P.BiasAdd()
Biasaddgrad = G.BiasAddGrad()
Cast = P.Cast()
MatMul = P.MatMul()

Fusion_relu_relu = Primitive('FusionOp_ReLU_ReLU')
Fusion_biasadd = Primitive('FusionOp_ReLU_ReLU_ReLU_BiasAdd_ReLU_ReLU_ReLU')
Fusion_biasaddgrad = Primitive('FusionOp_ReLU_ReLU_ReLU_BiasAddGrad_ReLU_ReLU_ReLU')
Fusion_matmul_relu = Primitive('FusionOp_MatMul_ReLU')

Add = P.Add()
Sub = P.Sub()
make_tuple = Primitive('MakeTuple')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_tbe_eltwise_fusion_1(tag):
    fns = FnDict()

    @fns
    def before(x):
        relu1 = Relu(x)
        relu2 = Relu(relu1)
        res = Cast(relu2, mstype.float16)
        return res

    @fns
    def after(x):
        fusion = Fusion_relu_relu(x)
        res = Cast(fusion)
        output = make_tuple(res)
        return output

    return fns[tag]


def test_tbe_eltwise_fusion_2(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        relu1 = Relu(x)
        relu2 = Relu(relu1)
        relu3 = Relu(relu2)
        biasadd = Biasadd(relu3, y)
        relu4 = Relu(biasadd)
        relu5 = Relu(relu4)
        relu6 = Relu(relu5)
        res = Cast(relu6, mstype.float16)
        return res

    @fns
    def after(x, y):
        fusion = Fusion_biasadd(x, y)
        res = Cast(fusion)
        output = make_tuple(res)
        return output

    return fns[tag]


def test_tbe_reduce_eltwise_fusion(tag):
    fns = FnDict()

    @fns
    def before(x):
        relu1 = Relu(x)
        relu2 = Relu(relu1)
        relu3 = Relu(relu2)
        biasaddgrad = Biasaddgrad(relu3)
        relu4 = Relu(biasaddgrad)
        relu5 = Relu(relu4)
        relu6 = Relu(relu5)
        res = Cast(relu6, mstype.float16)
        return res

    @fns
    def after(x):
        fusion = Fusion_biasaddgrad(x)
        res = Cast(fusion)
        output = make_tuple(res)
        return output

    return fns[tag]


def test_conv_singlein_fusion(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        conv = Conv(x, y)
        relu = Relu(conv)
        res = Cast(relu, mstype.float16)
        return res

    @fns
    def after(x, y):
        fusion = Fusion(x, y)
        res = Cast(fusion)
        output = make_tuple(res)
        return output

    return fns[tag]


def test_tbe_matmul_eltwise_fusion(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        matmul = MatMul(x, y)
        relu = Relu(matmul)
        res = Cast(relu, mstype.float16)
        return res

    @fns
    def after(x, y):
        fusion = Fusion_matmul_relu(x, y)
        res = Cast(fusion)
        output = make_tuple(res)
        return output

    return fns[tag]
