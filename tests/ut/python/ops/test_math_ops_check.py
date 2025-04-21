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
""" test ops """
import functools
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception, \
    pipeline_for_compile_forward_ge_graph_for_case_by_case_config


class AssignAddNet(nn.Cell):
    def __init__(self,):
        super(AssignAddNet, self).__init__()
        self.op = P.AssignAdd()
        self.inputdata = Parameter(Tensor(np.zeros([1]).astype(np.bool_), mstype.bool_), name="assign_add1")

    def construct(self, x):
        self.op(self.inputdata, x)
        return self.inputdata


class AssignSubNet(nn.Cell):
    def __init__(self,):
        super(AssignSubNet, self).__init__()
        self.op = P.AssignSub()
        self.inputdata = Parameter(Tensor(np.zeros([1]).astype(np.bool_), mstype.bool_), name="assign_sub1")

    def construct(self, x):
        self.op(self.inputdata, x)
        return self.inputdata


class ReduceNet(nn.Cell):
    def __init__(self, op_class, keep_dims, axis):
        super(ReduceNet, self).__init__()
        self.axis = axis
        self.op = op_class(keep_dims=keep_dims)

    def construct(self, x):
        return self.op(x, self.axis)


class CumProdNet(nn.Cell):
    def __init__(self):
        super(CumProdNet, self).__init__()
        self.op = P.CumProd()

    def construct(self, x, axis):
        return self.op(x, axis)


class CumSumNet(nn.Cell):
    def __init__(self, axis):
        super(CumSumNet, self).__init__()
        self.axis = axis
        self.op = P.CumSum()

    def construct(self, x):
        return self.op(x, self.axis)


raise_set = [
    # input two tensors, their shapes do not match
    ('TensorAdd2', {
        'block': (P.Add(), {'exception': ValueError, 'error_keywords': ['Add']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # check input Tensor(bool_)
    ('AssignAdd', {
        'block': (AssignAddNet(), {'exception': TypeError, 'error_keywords': ['AssignAdd']}),
        'desc_inputs': [Tensor(np.ones([1]).astype(np.bool_), mstype.bool_)],
        'skip': ['backward']}),

    # check input Tensor(bool_)
    ('AssignSub', {
        'block': (AssignSubNet(), {'exception': TypeError, 'error_keywords': ['AssignSub']}),
        'desc_inputs': [Tensor(np.ones([1]).astype(np.bool_), mstype.bool_)],
        'skip': ['backward']}),

    # type of axis is float, not int
    ('ReduceMean1', {
        'block': (ReduceNet(P.ReduceMean, keep_dims=True, axis=5.0),
                  {'exception': TypeError, 'error_keywords': ['ReduceMean']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),
    # axis is out of range
    ('ReduceMean2', {
        'block': (ReduceNet(P.ReduceMean, keep_dims=True, axis=5),
                  {'exception': ValueError, 'error_keywords': ['ReduceMean']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),

    # type of axis is float, not int
    ('ReduceSum1', {
        'block': (ReduceNet(P.ReduceSum, keep_dims=True, axis=5.0),
                  {'exception': TypeError, 'error_keywords': ['ReduceSum']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),
    # axis is out of range
    ('ReduceSum2', {
        'block': (ReduceNet(P.ReduceSum, keep_dims=True, axis=5),
                  {'exception': ValueError, 'error_keywords': ['ReduceSum']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),

    # type of axis is float, not int
    ('ReduceAll1', {
        'block': (ReduceNet(P.ReduceAll, keep_dims=True, axis=5.0),
                  {'exception': TypeError, 'error_keywords': ['ReduceAll']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.bool_))],
        'skip': ['backward']}),
    # axis is out of range
    ('ReduceAll2', {
        'block': (ReduceNet(P.ReduceAll, keep_dims=True, axis=5),
                  {'exception': ValueError, 'error_keywords': ['ReduceAll']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.bool_))],
        'skip': ['backward']}),

    # type of axis is float, not int
    ('ReduceMax1', {
        'block': (ReduceNet(P.ReduceMax, keep_dims=True, axis=5.0),
                  {'exception': TypeError, 'error_keywords': ['ReduceMax']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),
    # axis is out of range
    ('ReduceMax2', {
        'block': (ReduceNet(P.ReduceMax, keep_dims=True, axis=5),
                  {'exception': ValueError, 'error_keywords': ['ReduceMax']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),

    # type of axis is float, not int
    ('ReduceMin1', {
        'block': (ReduceNet(P.ReduceMin, keep_dims=True, axis=5.0),
                  {'exception': TypeError, 'error_keywords': ['ReduceMin']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),
    # axis is out of range
    ('ReduceMin2', {
        'block': (ReduceNet(P.ReduceMin, keep_dims=True, axis=5),
                  {'exception': ValueError, 'error_keywords': ['ReduceMin']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),

    # type of axis is float, not int
    ('ReduceProd1', {
        'block': (ReduceNet(P.ReduceProd, keep_dims=True, axis=5.0),
                  {'exception': TypeError, 'error_keywords': ['ReduceProd']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),
    # axis is out of range
    ('ReduceProd2', {
        'block': (ReduceNet(P.ReduceProd, keep_dims=True, axis=5),
                  {'exception': ValueError, 'error_keywords': ['ReduceProd']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32))],
        'skip': ['backward']}),

    # type of x is Tensor(bool)
    ('CumProd1', {
        'block': (CumProdNet(),
                  {'exception': TypeError, 'error_keywords': ['CumProd']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.bool)), 1],
        'skip': ['backward']}),
    # type of axis in float, not int
    ('CumProd2', {
        'block': (CumProdNet(),
                  {'exception': TypeError, 'error_keywords': ['CumProd']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.float32)), 5.0],
        'skip': ['backward']}),

    # type of x and y are Tensor(uint32)
    ('MatMul1', {
        'block': (P.MatMul(),
                  {'exception': TypeError, 'error_keywords': ['MatMul']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.uint32)), Tensor(np.ones([3, 2]).astype(np.uint32))],
        'skip': ['backward']}),
    # type of x and y not match
    ('MatMul2', {
        'block': (P.MatMul(),
                  {'exception': TypeError, 'error_keywords': ['MatMul']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.int32))],
        'skip': ['backward']}),
    # shape of x and y not match
    ('MatMul3', {
        'block': (P.MatMul(),
                  {'exception': ValueError, 'error_keywords': ['MatMul']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.float32)), Tensor(np.ones([2, 3]).astype(np.float32))],
        'skip': ['backward']}),

    # dims of x and y are less than 3
    ('BatchMatMul1', {
        'block': (P.BatchMatMul(),
                  {'exception': ValueError, 'error_keywords': ['BatchMatMul']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.int32)), Tensor(np.ones([3, 2]).astype(np.int32))],
        'skip': ['backward']}),

    # type of x is Tensor(bool)
    ('CumSum1', {
        'block': (CumSumNet(axis=1),
                  {'exception': TypeError, 'error_keywords': ['CumSum']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.bool))],
        'skip': ['backward']}),
    # type of axis in float, not int
    ('CumSum2', {
        'block': (CumSumNet(axis=1.0),
                  {'exception': TypeError, 'error_keywords': ['CumSum']}),
        'desc_inputs': [Tensor(np.ones([2, 3, 5]).astype(np.bool))],
        'skip': ['backward']}),

    # intput is not tuple or list
    ('AddN1', {
        'block': (P.AddN(),
                  {'exception': TypeError, 'error_keywords': ['AddN']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.uint32))],
        'skip': ['backward']}),
    # type not match
    ('AddN2', {
        'block': (P.AddN(),
                  {'exception': TypeError, 'error_keywords': ['AddN']}),
        'desc_inputs': [(Tensor(np.ones([2, 3]).astype(np.uint32)), Tensor(np.ones([3, 2]).astype(np.int32)))],
        'skip': ['backward']}),
    # shape not match
    ('AddN3', {
        'block': (P.AddN(),
                  {'exception': ValueError, 'error_keywords': ['AddN']}),
        'desc_inputs': [(Tensor(np.ones([2, 3]).astype(np.int32)), Tensor(np.ones([3, 2]).astype(np.int32)))],
        'skip': ['backward']}),

    # input is Tensor(bool)
    ('Neg1', {
        'block': (P.Neg(),
                  {'exception': TypeError, 'error_keywords': ['Neg']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.bool_))],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('Sub2', {
        'block': (P.Sub(), {'exception': ValueError, 'error_keywords': ['Sub']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('Mul2', {
        'block': (P.Mul(), {'exception': ValueError, 'error_keywords': ['Mul']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input is Tensor(bool)
    ('Square1', {
        'block': (P.Square(),
                  {'exception': TypeError, 'error_keywords': ['Square']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is Tensor(bool)
    ('Rsqrt1', {
        'block': (P.Rsqrt(),
                  {'exception': TypeError, 'error_keywords': ['Rsqrt']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is Tensor(bool)
    ('Sqrt1', {
        'block': (P.Sqrt(),
                  {'exception': TypeError, 'error_keywords': ['Sqrt']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is not Tensor
    ('Reciprocal1', {
        'block': (P.Reciprocal(),
                  {'exception': TypeError, 'error_keywords': ['Reciprocal']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),

    # input is not Tensor
    ('Exp1', {
        'block': (P.Exp(),
                  {'exception': TypeError, 'error_keywords': ['Exp']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),

    # input is not Tensor
    ('Log1', {
        'block': (P.Log(),
                  {'exception': TypeError, 'error_keywords': ['Log']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('Minimum2', {
        'block': (P.Minimum(), {'exception': ValueError, 'error_keywords': ['Minimum']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('Maximum2', {
        'block': (P.Maximum(), {'exception': ValueError, 'error_keywords': ['Maximum']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('RealDiv2', {
        'block': (P.RealDiv(), {'exception': ValueError, 'error_keywords': ['RealDiv']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('Div2', {
        'block': (P.Div(), {'exception': ValueError, 'error_keywords': ['Div']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('FloorDiv2', {
        'block': (P.FloorDiv(), {'exception': ValueError, 'error_keywords': ['FloorDiv']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input x is Tensor(int32), not Tensor(float)
    ('Floor1', {
        'block': (P.Floor(),
                  {'exception': TypeError, 'error_keywords': ['Floor']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.int32))],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('FFloorMod2', {
        'block': (P.FloorMod(), {'exception': ValueError, 'error_keywords': ['FloorMod']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input x is Tensor(int32), not Tensor(float)
    ('Acosh1', {
        'block': (P.Acosh(),
                  {'exception': TypeError, 'error_keywords': ['Acosh']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.bool_))],
        'skip': ['backward']}),

    # shape of x and y not match
    ('Equal2', {
        'block': (P.Equal(), {'exception': ValueError, 'error_keywords': ['Equal']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32))],
        'skip': ['backward']}),

    # input is not tensor
    ('EqualCount0', {
        'block': (P.EqualCount(), {'exception': TypeError, 'error_keywords': ['EqualCount']}),
        'desc_inputs': [5.0, Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # type of x and y not match
    ('EqualCount1', {
        'block': (P.EqualCount(), {'exception': TypeError, 'error_keywords': ['EqualCount']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # shape of x and y not match

    # shape of x and y not match
    ('NotEqual2', {
        'block': (P.NotEqual(), {'exception': ValueError, 'error_keywords': ['NotEqual']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32))],
        'skip': ['backward']}),

    # shape of x and y not match
    ('Greater2', {
        'block': (P.Greater(), {'exception': ValueError, 'error_keywords': ['Greater']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32))],
        'skip': ['backward']}),

    # shape of x and y not match
    ('GreaterEqual2', {
        'block': (P.GreaterEqual(), {'exception': ValueError, 'error_keywords': ['GreaterEqual']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32))],
        'skip': ['backward']}),

    # shape of x and y not match
    ('Less2', {
        'block': (P.Less(), {'exception': ValueError, 'error_keywords': ['Less']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32))],
        'skip': ['backward']}),

    # shape of x and y not match
    ('LessEqual2', {
        'block': (P.LessEqual(), {'exception': ValueError, 'error_keywords': ['LessEqual']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), Tensor(np.ones([3, 2]).astype(np.float32))],
        'skip': ['backward']}),

    # input x is not Tensor(bool)
    ('LogicalNot1', {
        'block': (P.LogicalNot(),
                  {'exception': TypeError, 'error_keywords': ['LogicalNot']}),
        'desc_inputs': [Tensor(np.ones([2, 3]).astype(np.int32))],
        'skip': ['backward']}),

    # type of x and y not match
    ('LogicalAnd1', {
        'block': (P.LogicalAnd(), {'exception': TypeError, 'error_keywords': ['LogicalAnd']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),
    # shape of x and y not match
    ('LogicalAnd2', {
        'block': (P.LogicalAnd(), {'exception': ValueError, 'error_keywords': ['LogicalAnd']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_)), Tensor(np.ones([3, 2]).astype(np.bool_))],
        'skip': ['backward']}),

    # type of x and y not match
    ('LogicalOr1', {
        'block': (P.LogicalOr(), {'exception': TypeError, 'error_keywords': ['LogicalOr']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),
    # shape of x and y not match
    ('LogicalOr2', {
        'block': (P.LogicalOr(), {'exception': ValueError, 'error_keywords': ['LogicalOr']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_)), Tensor(np.ones([3, 2]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is not tensor
    ('NPUGetFloatStatus0', {
        'block': (P.NPUGetFloatStatus(), {'exception': TypeError, 'error_keywords': ['NPUGetFloatStatus']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(int32), not Tensor(float32)
    ('NPUGetFloatStatus1', {
        'block': (P.NPUGetFloatStatus(), {'exception': TypeError, 'error_keywords': ['NPUGetFloatStatus']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32))],
        'skip': ['backward']}),
    # dims is not 1
    ('NPUGetFloatStatus2', {
        'block': (P.NPUGetFloatStatus(), {'exception': ValueError, 'error_keywords': ['NPUGetFloatStatus']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # shape[0] is not 8
    ('NPUGetFloatStatus3', {
        'block': (P.NPUGetFloatStatus(), {'exception': ValueError, 'error_keywords': ['NPUGetFloatStatus']}),
        'desc_inputs': [Tensor(np.ones([3]).astype(np.float32))],
        'skip': ['backward']}),

    # input is not tensor
    ('NPUClearFloatStatus0', {
        'block': (P.NPUClearFloatStatus(), {'exception': TypeError, 'error_keywords': ['NPUClearFloatStatus']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(int32), not Tensor(float32)
    ('NPUClearFloatStatus1', {
        'block': (P.NPUClearFloatStatus(), {'exception': TypeError, 'error_keywords': ['NPUClearFloatStatus']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32))],
        'skip': ['backward']}),
    # dims is not 1
    ('NPUClearFloatStatus2', {
        'block': (P.NPUClearFloatStatus(), {'exception': ValueError, 'error_keywords': ['NPUClearFloatStatus']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # shape[0] is not 8
    ('NPUClearFloatStatus3', {
        'block': (P.NPUClearFloatStatus(), {'exception': ValueError, 'error_keywords': ['NPUClearFloatStatus']}),
        'desc_inputs': [Tensor(np.ones([3]).astype(np.float32))],
        'skip': ['backward']}),

    # input is not tensor
    ('Cos0', {
        'block': (P.Cos(), {'exception': TypeError, 'error_keywords': ['Cos']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('Cos1', {
        'block': (P.Cos(), {'exception': TypeError, 'error_keywords': ['Cos']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is not tensor
    ('ACos0', {
        'block': (P.ACos(), {'exception': TypeError, 'error_keywords': ['ACos']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('ACos1', {
        'block': (P.ACos(), {'exception': TypeError, 'error_keywords': ['ACos']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is not tensor
    ('Sin0', {
        'block': (P.Sin(), {'exception': TypeError, 'error_keywords': ['Sin']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('Sin1', {
        'block': (P.Sin(), {'exception': TypeError, 'error_keywords': ['Sin']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is not tensor
    ('NMSWithMask0', {
        'block': (P.NMSWithMask(), {'exception': TypeError, 'error_keywords': ['NMSWithMask']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is not Tensor(float16) or Tensor(float32)
    ('NMSWithMask1', {
        'block': (P.NMSWithMask(), {'exception': TypeError, 'error_keywords': ['NMSWithMask']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32))],
        'skip': ['backward']}),
    # dims is not 2
    ('NMSWithMask2', {
        'block': (P.NMSWithMask(), {'exception': ValueError, 'error_keywords': ['NMSWithMask']}),
        'desc_inputs': [Tensor(np.ones([3, 4, 2]).astype(np.float32))],
        'skip': ['backward']}),
    # shape[1] is not 5
    ('NMSWithMask3', {
        'block': (P.NMSWithMask(), {'exception': ValueError, 'error_keywords': ['NMSWithMask']}),
        'desc_inputs': [Tensor(np.ones([3, 2]).astype(np.float32))],
        'skip': ['backward']}),

    # input is not tensor
    ('Abs0', {
        'block': (P.Abs(), {'exception': TypeError, 'error_keywords': ['Abs']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('Abs1', {
        'block': (P.Abs(), {'exception': TypeError, 'error_keywords': ['Abs']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is not tensor
    ('Sign0', {
        'block': (P.Sign(), {'exception': TypeError, 'error_keywords': ['Sign']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('Sign1', {
        'block': (P.Sign(), {'exception': TypeError, 'error_keywords': ['Sign']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is not tensor
    ('Round0', {
        'block': (P.Round(), {'exception': TypeError, 'error_keywords': ['Round']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('Round1', {
        'block': (P.Round(), {'exception': TypeError, 'error_keywords': ['Round']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),

    # input two tensors, their shapes do not match
    ('Atan22', {
        'block': (P.Atan2(), {'exception': ValueError, 'error_keywords': ['Atan2']}),
        'desc_inputs': [Tensor(np.ones([3, 5]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
]

test_case_math_ops = [
    # input two tensors, but element types are not same
    ('TensorAdd1', {
        'block': P.Add(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('Sub1', {
        'block': P.Sub(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('Mul1', {
        'block': P.Mul(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('Minimum1', {
        'block': P.Minimum(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('Maximum1', {
        'block': P.Maximum(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('RealDiv1', {
        'block': P.RealDiv(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('Div1', {
        'block': P.Div(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('FloorDiv1', {
        'block': P.FloorDiv(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('FloorMod1', {
        'block': P.FloorMod(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # type of x and y not match
    ('Equal1', {
        'block': P.Equal(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # type of x and y not match
    ('NotEqual1', {
        'block': P.NotEqual(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # type of x and y not match
    ('Greater1', {
        'block': P.Greater(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # type of x and y not match
    ('GreaterEqual1', {
        'block': P.GreaterEqual(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # type of x and y not match
    ('Less1', {
        'block': P.Less(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # type of x and y not match
    ('LessEqual1', {
        'block': P.LessEqual(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input two tensors, but element types are not same
    ('Atan21', {
        'block': P.Atan2(),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32)), Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
]


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception)
def test_check_exception():
    return raise_set


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    import mindspore.context as context
    context.set_context(mode=context.GRAPH_MODE)
    return functools.reduce(lambda x, y: x + y, [test_case_math_ops])
