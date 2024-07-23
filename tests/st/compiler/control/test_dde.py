# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np
from tests.st.compiler.control.cases_register import case_register
import mindspore.context as context
import mindspore as ms
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import ops, Tensor, nn

context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_dde_make_tuple_joined_with_tuple_output_primitive():
    """
    Feature: Eliminate unused element for tuple.
    Description: Two branch return make tuple and tuple output node like top_k
    Expectation: Correct result and no exception.
    """

    @ms.jit
    def topk_fun(x, k):
        if k == 0:
            output = (ms.ops.ones((0,), dtype=ms.float32), ms.ops.ones((0,), dtype=ms.int32))
        else:
            output = ms.ops.topk(x, k, None, True, True)
        return output

    x = ms.tensor([1., 2., 3.])
    k = ms.tensor([0])
    out = topk_fun(x, k)
    expect_out0 = ms.ops.ones((0,), dtype=ms.float32)
    expect_out1 = ms.ops.ones((0,), dtype=ms.int32)
    assert np.allclose(out[0].asnumpy(), expect_out0.asnumpy())
    assert np.allclose(out[1].asnumpy(), expect_out1.asnumpy())


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_dde_parameter_converted_to_value_tuple():
    """
    Feature: Eliminate unused element for tuple.
    Description: The value_tuple is converted from the parameter which is not set sequence_nodes.
    Expectation: Correct result and no exception.
    """

    def _old_norm(norm_type, x):
        out = F.pow((F.reduce_sum(F.pow(x, norm_type))), 1. / norm_type).astype(x.dtype)
        return out

    class ClipByNormFuncNet(nn.Cell):
        def __init__(self, max_norm, norm_type=2.0, error_if_nonfinite=False):
            super().__init__()
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.error_if_nonfinite = error_if_nonfinite
            self.partial_op = P.Partial()
            self.hyper_map = C.HyperMap()

        def construct(self, *x):
            is_tensor = False
            if isinstance(x, Tensor):
                x = [x]
                is_tensor = True
            total_norm = _old_norm(self.norm_type,
                                   F.stack(self.hyper_map(self.partial_op(_old_norm, self.norm_type), x)))
            clip_coef = self.max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                ret = self.hyper_map(self.partial_op(F.mul, clip_coef), x)
            else:
                ret = x
            if is_tensor:
                return ret[0]
            return ret

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation(sens_param=True)

        def construct(self, *x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(*x)

    ms.set_context(mode=ms.GRAPH_MODE)
    net = ClipByNormFuncNet(max_norm=1, norm_type=2, error_if_nonfinite=True)
    net.set_train()
    x = [ops.ones((2, 2)), ops.ones((2,))]
    ms_output = net(*x)
    output = GradNetWrtX(net)(*x, ms_output)
    expect_out = np.array([[0.4082482, 0.4082482], [0.4082482, 0.4082482]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_out)
