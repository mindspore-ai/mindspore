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
""" renormalize_test """
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import ops
import mindspore as ms


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fn_dict.get(name)


# Test ut for function: Renormalize.
def test_renormalize(tag):
    """
    Feature: Renormalize.
    Description: test cases for renormalize.
    Expectation: graph can be renormalized and no exception raised.
    """
    fns = FnDict()
    x = Tensor([1], mstype.int32)
    y = Tensor([1, 2], mstype.int32)
    pow_ops = P.Pow()

    def test_poly_delay_specialize_ut():
        def function_h():
            pow_res = pow_ops(x, x)

            def function_g(param_x):
                return pow(pow_res, param_x)

            return F.make_tuple(pow_res, function_g)

        def function_f():
            h_out = function_h()
            h_forward_out = F.tuple_getitem(h_out, 0)
            g = F.tuple_getitem(h_out, 1)

            def function_k():
                kout1 = g(x)
                kout2 = g(y)
                kout = F.depend(kout1, kout2)
                return kout

            return F.make_tuple(h_forward_out, function_k)

        out = function_f()
        forward_out = F.tuple_getitem(out, 0)
        closure_out = F.tuple_getitem(out, 1)
        closure_out_tensor = closure_out()
        return F.add(forward_out, closure_out_tensor)
    ######################################################

    def test_ignore_flag_with_twice_call_if():
        ll = (Tensor([1]), Tensor([2]), Tensor([3]))

        def func(index, m, n):
            func_out = Tensor([0])
            if ops.less(m, n):
                func_out = ms.ops.tuple_getitem(ll, index)
            return func_out
        x1 = pow_ops(x, x)
        y1 = pow_ops(x, x)
        out1 = func(0, x1, y1)
        out2 = func(1, x1, y1)
        return out1, out2
    ######################################################
    # Add test_poly_delay_specialize_ut to fn dict.
    fns(test_poly_delay_specialize_ut)
    fns(test_ignore_flag_with_twice_call_if)

    return fns[tag]
