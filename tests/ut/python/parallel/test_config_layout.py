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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, dp=None, sp=None, mp=None):
        super().__init__()
        self.transpose_q = P.Transpose().shard(((dp, sp, mp, 1),))
        self.transpose_k = P.Transpose().shard(((dp, sp, mp, 1),))
        self.transpose_v = P.Transpose().shard(((dp, sp, mp, 1),))

        self.real_div = P.RealDiv().shard(((dp, mp, sp, 1), ()))
        self.real_div.add_prim_attr("layout", {"dev_matrix": (dp, sp, mp, 1),
                                               "input_tensor_map": ((3, 1, 2, 0), ())})

        self.qk_matmul = P.BatchMatMul().shard(((dp, mp, sp, 1), (dp, mp, 1, 1)))
        self.qk_matmul.add_prim_attr("layout", {"dev_matrix": (dp, sp, mp, 1),
                                                "input_tensor_map": ((3, 1, 2, 0), (3, 1, -1, -1))})

        self.cast = P.Cast()

        self.add = P.Add().shard(((dp, 1, sp, 1), (dp, mp, sp, 1)))
        self.add.add_prim_attr("layout", {"dev_matrix": (dp, sp, mp, 1),
                                          "input_tensor_map": ((3, -1, 2, 0), (3, 1, 2, 0))})

        self.soft_max = P.Softmax().shard(((dp, mp, sp, 1),))
        self.soft_max.add_prim_attr("layout", {"dev_matrix": (dp, sp, mp, 1),
                                               "input_tensor_map": ((3, 1, 2, 0),)})

        self.v_matmul = P.BatchMatMul().shard(((dp, mp, sp, 1), (dp, mp, 1, 1)))
        self.v_matmul.add_prim_attr("layout", {"dev_matrix": (dp, sp, mp, 1),
                                               "input_tensor_map": ((3, 1, 2, 0), (3, 1, -1, -1))})

        self.merger_head_transpose = P.Transpose().shard(((dp, mp, sp, 1),))
        self.merger_head_transpose.add_prim_attr("layout", {"dev_matrix": (dp, sp, mp, 1),
                                                            "input_tensor_map": ((3, 1, 2, 0),)})

        self.reshape = P.Reshape()
        self.reshape.add_prim_attr("skip_redistribution", True)

        self.linear = P.MatMul().shard(((dp * sp, mp), (mp, 1)))
        self.param = Parameter(Tensor(np.ones([5120, 5120]), dtype=ms.float16), "w1")

    def construct(self, q, k, v, x):
        q = self.transpose_q(q, (0, 2, 1, 3))
        q = self.real_div(q, Tensor(2.0, dtype=q.dtype))
        k = self.transpose_k(k, (0, 2, 3, 1))
        score = self.qk_matmul(q, k)
        score = self.cast(score, ms.float32)
        attention_scores = self.add(x, score)
        attention_probs = self.soft_max(attention_scores)
        attention_probs = self.cast(attention_probs, ms.float16)
        v = self.transpose_v(v, (0, 2, 1, 3))
        weighted_values = self.v_matmul(attention_probs, v)
        weighted_values_transpose = self.merger_head_transpose(weighted_values, (0, 2, 1, 3))
        out = self.reshape(weighted_values_transpose, (-1, 5120))
        out = self.linear(out, self.param)
        return out


def test_layout_attr():
    """
    Feature: test config layout attr for primitive
    Description: used for seq parallel
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)

    net = Net(dp=2, sp=2, mp=2)
    q = Tensor(np.ones([8, 1024, 40, 128]), dtype=ms.float16)
    k = Tensor(np.ones([8, 1024, 40, 128]), dtype=ms.float16)
    v = Tensor(np.ones([8, 1024, 40, 128]), dtype=ms.float16)
    x = Tensor(np.ones([8, 1, 1024, 1024]), dtype=ms.float16)
    phase = compile_net(net, q, k, v, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('RealDiv-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('BatchMatMul-0', ['RealDiv-0'])
    assert validator.check_node_inputs_has('Softmax-0', ['Add-0'])
    assert validator.check_node_inputs_has('MatMul-0', ['Reshape'])
    assert validator.check_parameter_shape('w1', [2560, 5120])
