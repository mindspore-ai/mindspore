# Copyright 2024 Huawei Technologies Co., Ltd
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

"""
sync bn op test case
"""

import numpy as np
from mindspore import ops
from mindspore import context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from tests.st.graph_kernel.gk_utils import AssertGKEnable
from tests.mark_utils import arg_mark


def get_output(net_cls, inputs, op_name, inplace_idx, enable_graph_kernel):
    context.set_context(mode=context.GRAPH_MODE)
    jit_level = "O1" if enable_graph_kernel else "O0"
    context.set_context(jit_config={"jit_level": jit_level})
    if enable_graph_kernel:
        context.set_context(graph_kernel_flags="--enable_expand_ops={}".format(op_name))
    with AssertGKEnable(enable_graph_kernel):
        net = net_cls()
        outputs = net(*inputs)
    outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    for idx in inplace_idx:
        outputs.append(inputs[idx])
    outputs = [output.asnumpy() for output in outputs]
    return outputs


def compare(expects, outputs):
    for e, o in zip(expects, outputs):
        np.testing.assert_allclose(e, o, 1e-4, 1e-4)


def run_batch_norm_stats():
    class Net(Cell):
        def construct(self, x0, eps):
            return ops.auto_generate.BatchNormStats()(x0, eps)

    inputs = [Tensor(np.random.normal(0, 1, (30, 24, 17, 32)).astype(np.float32)), 1e-5]
    outputs = get_output(Net, inputs, "BatchNormStats", [], True)
    expects = get_output(Net, inputs, "BatchNormStats", [], False)
    compare(expects, outputs)


def run_batch_gather_stats_with_counts():
    class Net(Cell):
        def construct(self, x0, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all):
            return ops.auto_generate.BatchNormGatherStatsWithCounts()(x0, mean_all, invstd_all, running_mean,
                                                                      running_var, momentum, eps, count_all.view(-1))

    inputs = [np.random.normal(0, 1, (30, 24, 17, 32)).astype(np.float32),
              np.random.normal(0, 1, (8, 24)).astype(np.float32),
              np.abs(np.random.normal(0, 1, (8, 24)).astype(np.float32)) + 1e-5,
              np.random.normal(0, 1, (24,)).astype(np.float32),
              np.random.normal(0, 1, (24,)).astype(np.float32),
              0.1, 1e-5,
              np.array([16320, 16320, 16320, 16320, 16320, 16320, 16320, 14144]).astype(np.float32)]
    inputs_ms = [Tensor(inputs[0]), Tensor(inputs[1]), Tensor(inputs[2]),
                 Parameter(Tensor(np.array(inputs[3])), name="p0"),
                 Parameter(Tensor(np.array(inputs[4])), name="p1"),
                 inputs[5], inputs[6], Tensor(inputs[7])]
    outputs = get_output(Net, inputs_ms, "BatchNormGatherStatsWithCounts", [], True)
    inputs_ms = [Tensor(inputs[0]), Tensor(inputs[1]), Tensor(inputs[2]),
                 Parameter(Tensor(np.array(inputs[3])), name="x0"),
                 Parameter(Tensor(np.array(inputs[4])), name="x1"),
                 inputs[5], inputs[6], Tensor(inputs[7])]
    expects = get_output(Net, inputs_ms, "BatchNormGatherStatsWithCounts", [], False)
    compare(expects, outputs)


def run_batch_norm_element():
    class Net(Cell):
        def construct(self, x0, weight, bias, mean, invstd, eps):
            return ops.auto_generate.BatchNormElemt()(x0, weight, bias, mean, invstd, eps)

    inputs = [Tensor(np.random.normal(0, 1, (30, 24, 17, 32)).astype(np.float32)),
              Tensor(np.random.normal(0, 1, (24,)).astype(np.float32)),
              Tensor(np.random.normal(0, 1, (24,)).astype(np.float32)),
              Tensor(np.random.normal(0, 1, (24,)).astype(np.float32)),
              Tensor(np.abs(np.random.normal(0, 1, (24,)).astype(np.float32)) + 1e-5),
              1e-5]
    outputs = get_output(Net, inputs, "BatchNormElemt", [], True)
    expects = get_output(Net, inputs, "BatchNormElemt", [], False)
    compare(expects, outputs)


def run_batch_norm_element_grad():
    class Net(Cell):
        def construct(self, dout, x, mean, invstd, weight, sumd_dy, sum_dy_xmu, count):
            return ops.auto_generate.BatchNormElemtGrad()(dout, x, mean, invstd, weight, sumd_dy, sum_dy_xmu, count)

    inputs = [Tensor(np.random.normal(0, 1, (30, 24, 17, 32)).astype(np.float32)),
              Tensor(np.random.normal(0, 1, (30, 24, 17, 32)).astype(np.float32)),
              Tensor(np.random.normal(0, 1, (24,)).astype(np.float32)),
              Tensor(np.abs(np.random.normal(0, 1, (24,)).astype(np.float32)) + 1e-5),
              Tensor(np.random.normal(0, 1, (24,)).astype(np.float32)),
              Tensor(np.random.normal(0, 1, (24,)).astype(np.float32)),
              Tensor(np.random.normal(0, 1, (24,)).astype(np.float32)),
              Tensor(np.array([16320, 16320, 16320, 16320, 16320, 16320, 16320, 14144]).astype(np.float32))]
    outputs = get_output(Net, inputs, "BatchNormElemtGrad", [], True)
    expects = get_output(Net, inputs, "BatchNormElemtGrad", [], False)
    compare(expects, outputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=14.00)
def test_sync_bn():
    """
    Feature: test sync bn op
    Description: sync bn op expander
    Expectation: the result match with the expected result
    """
    run_batch_norm_stats()
    run_batch_gather_stats_with_counts()
    run_batch_norm_element()
    run_batch_norm_element_grad()
