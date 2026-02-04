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

"""test hccl communication op fusion feature"""

import subprocess
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.communication.management import init
from mindspore import context

def find_file(file, para):
    output = subprocess.check_output(["grep '%s' ./%s|wc -l" % (para, file)], shell=True)
    return str(output, 'utf-8').strip()

np.random.seed(1)
context.set_context(jit_level='O0')
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
init()


class AllReduceFusionNet(nn.Cell):
    def __init__(self, is_fusion=False):
        super(AllReduceFusionNet, self).__init__()
        self.add_op_list = []
        self.allreduce_op_list = []
        for i in range(1):
            self.add_op_list.append(P.Add())
            all_reduce_op = P.AllReduce(P.ReduceOp.SUM)
            if is_fusion:
                all_reduce_op.add_prim_attr("fusion", i // 2 + 1)
            self.allreduce_op_list.append(all_reduce_op)

    def construct(self, x0):
        x0 = self.add_op_list[0](x0, x0)
        x0 = self.allreduce_op_list[0](x0)
        return x0


def test_hccl_allreduce_fusion_by_attr():
    """
    Feature: test 'AllReduce' op fusion.
    Description: test 'AllReduce' op fusion feature.
    Expectation: expect correct result and ir.
    """
    context.set_context(save_graphs=True)
    comm_fusion_dict = {"allreduce": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, comm_fusion=comm_fusion_dict)
    input_x = np.ones([3, 4]).astype(np.float32)

    net1 = AllReduceFusionNet(False)
    output1 = net1(Tensor(input_x, mstype.float32))
    ir_allreduce_num1 = find_file('rank_0/graph_build_0*.ir', ' AllReduce')
    assert ir_allreduce_num1 == '1'

    net2 = AllReduceFusionNet(False)
    output2 = net2(Tensor(input_x, mstype.float32))
    ir_allreduce_num2 = find_file('rank_0/graph_build_1*.ir', ' AllReduce')
    assert ir_allreduce_num2 == '1'

    expect_output = [input_x * 16]
    assert np.allclose(output1.asnumpy(), output2.asnumpy())
    assert np.allclose(expect_output, output1.asnumpy())
    assert output1.shape == (3, 4)


class AllGatherFusionNet(nn.Cell):
    def __init__(self, is_fusion=False):
        super(AllGatherFusionNet, self).__init__()
        self.add_op_list = []
        self.allgather_op_list = []
        for i in range(4):
            self.add_op_list.append(P.Add())
            all_gather_op = P.AllGather()
            if is_fusion:
                all_gather_op.add_prim_attr("fusion", i // 2 + 1)
            self.allgather_op_list.append(all_gather_op)

    def construct(self, x0, x1, x2, x3):
        x0 = self.add_op_list[0](x0, x0)
        x0 = self.allgather_op_list[0](x0)

        x1 = self.add_op_list[1](x1, x1)
        x1 = self.allgather_op_list[1](x1)

        x2 = self.add_op_list[2](x2, x2)
        x2 = self.allgather_op_list[2](x2)

        x3 = self.add_op_list[3](x3, x3)
        x3 = self.allgather_op_list[3](x3)
        return x0, x1, x2, x3


def test_hccl_allgather_fusion_by_attr():
    """
    Feature: test 'AllGather' op fusion.
    Description: test 'AllGather' op fusion feature.
    Expectation: expect correct result and ir.
    """
    context.set_context(save_graphs=True)
    comm_fusion_dict = {"allgather": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, comm_fusion=comm_fusion_dict)
    input_x = np.ones([1]).astype(np.float32)

    net1 = AllGatherFusionNet(False)
    output1 = net1(Tensor(input_x, mstype.float32), Tensor(input_x, mstype.float32),
                   Tensor(input_x, mstype.float32), Tensor(input_x, mstype.float32))
    ir_allgather_num1 = find_file('rank_0/graph_build_0*.ir', ' AllGather')
    assert ir_allgather_num1 == '4'

    net2 = AllGatherFusionNet(False)
    output2 = net2(Tensor(input_x, mstype.float32), Tensor(input_x, mstype.float32),
                   Tensor(input_x, mstype.float32), Tensor(input_x, mstype.float32))
    ir_allgather_num2 = find_file('rank_0/graph_build_1*.ir', ' AllGather')
    assert ir_allgather_num2 == '4'

    expect_output = [[2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2],
                     [2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2]]
    for i in range(4):
        assert np.allclose(output1[i].asnumpy(), output2[i].asnumpy())
        assert np.allclose(expect_output, output1[i].asnumpy())
        assert output1[i].asnumpy().shape == (8,)


class ReduceScatterFusionNet(nn.Cell):
    def __init__(self, is_fusion=False):
        super(ReduceScatterFusionNet, self).__init__()
        self.add_op_list = []
        self.reducescatter_op_list = []
        for i in range(4):
            self.add_op_list.append(P.Add())
            reduce_scatter_op = P.ReduceScatter()
            if is_fusion:
                reduce_scatter_op.add_prim_attr("fusion", i // 2 + 1)
            self.reducescatter_op_list.append(reduce_scatter_op)

    def construct(self, x0, x1, x2, x3):
        x0 = self.add_op_list[0](x0, x0)
        x0 = self.reducescatter_op_list[0](x0)

        x1 = self.add_op_list[1](x1, x1)
        x1 = self.reducescatter_op_list[1](x1)

        x2 = self.add_op_list[2](x2, x2)
        x2 = self.reducescatter_op_list[2](x2)

        x3 = self.add_op_list[3](x3, x3)
        x3 = self.reducescatter_op_list[3](x3)
        return x0, x1, x2, x3


def test_hccl_reducescatter_fusion_by_attr():
    """
    Feature: test 'ReduceScatter' op fusion.
    Description: test 'ReduceScatter' op fusion feature.
    Expectation: expect correct result and ir.
    """
    context.set_context(save_graphs=True)
    comm_fusion_dict = {"reducescatter": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, comm_fusion=comm_fusion_dict)
    input_x = np.ones([8]).astype(np.float32)

    net1 = ReduceScatterFusionNet(False)
    output1 = net1(Tensor(input_x, mstype.float32), Tensor(input_x, mstype.float32),
                   Tensor(input_x, mstype.float32), Tensor(input_x, mstype.float32))
    ir_reducescatter_num1 = find_file('rank_0/graph_build_0*.ir', ' ReduceScatter')
    assert ir_reducescatter_num1 == '4'

    net2 = ReduceScatterFusionNet(False)
    output2 = net2(Tensor(input_x, mstype.float32), Tensor(input_x, mstype.float32),
                   Tensor(input_x, mstype.float32), Tensor(input_x, mstype.float32))
    ir_reducescatter_num2 = find_file('rank_0/graph_build_1*.ir', ' ReduceScatter')
    assert ir_reducescatter_num2 == '4'

    expect_output = [[16], [16], [16], [16]]
    for i in range(4):
        assert np.allclose(output1[i].asnumpy(), output2[i].asnumpy())
        assert np.allclose(expect_output, output1[i].asnumpy())
        assert output1[i].asnumpy().shape == (1,)
