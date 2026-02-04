# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test high-dimensional batch matrix multiplication with N-dimensional tensor parallelism.

This module tests the BatchMatMul operation in MindSpore with high-dimensional tensor
support and N-dimensional tensor parallelism (ND-TP) optimization enabled. It validates
the distributed execution of batch matrix multiplication across multiple devices with
complex sharding strategies.

Testing Workflow:
1. Initialize distributed training environment with 8 devices
2. Configure semi-automatic parallel mode with Layout-based sharding
3. Create BatchMatMul network with ND-TP optimization enabled
4. Set up 4D tensor layout (batch, x, y, z dimensions) with 2x2x2x1 device mesh
5. Define input/parameter sharding strategies using layout descriptors
6. Create training dataset and model
7. Execute training with dataset_sink_mode=False for debugging
8. Validate that RuntimeError is properly raised for unsupported configurations

Key Components:
- BatchMatMulCell: Network cell performing batch matrix multiplication with configurable strategies
- in_layout: 4D Layout object defining tensor dimension mapping to device mesh
- in_strategy: Tuple of two Layout-based sharding strategies for input and parameter
- enable_nd_tp: Boolean attribute enabling N-dimensional tensor parallelism optimization

Test Configuration:
- Device count: 8 (arranged as 2x2x2x1 mesh)
- Tensor shape: (128, 96, 128) for batch matrix multiplication
- Dataset: FakeData with 256 samples, batch size 16
- Distributed sharding: Semi-automatic parallel mode with custom strategies
"""
import mindspore as ms
import numpy as np
import pytest
from mindspore import nn, Parameter, context, Tensor
import mindspore.ops.operations as P
from mindspore.communication.management import init
from mindspore.parallel.shard import Layout
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from tests.st.auto_parallel.model_parallel import FakeData
from tests.st.auto_parallel.utils.modeltrain_base import modeltrainbase


def setup_function():
    ms.set_context(mode=0)
    ms.set_context(save_graphs=True, save_graphs_path="./parallel_batchmatmul_high_dim/ir")
    init()


class BatchMatMulCell(nn.Cell):
    def __init__(self, in_strategy, parameter_shape, transpose_a=False, transpose_b=False,
                 out_strategy=None):
        super().__init__()
        matmul_np = np.full(parameter_shape, 0.5, dtype=np.float32)
        self.param = Parameter(Tensor(matmul_np), name="matmul_weight")
        if out_strategy is None:
            self.batchmatmul = P.BatchMatMul(transpose_a, transpose_b).shard(
                in_strategy=in_strategy)
        else:
            self.batchmatmul = P.BatchMatMul(transpose_a, transpose_b).shard(
                in_strategy=in_strategy, out_strategy=out_strategy)

    def construct(self, x, label):
        out = self.batchmatmul(x, self.param)
        return out



def test_parallel_batchmatmul_enable_nd_tp_002():
    D.init()
    rank_id = get_rank()
    # 分布式执行训练，返回最新的ckpt
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8)
    in_layout = Layout((2, 2, 2, 1), ("x", "y", "z", "b"))
    in_strategy = (in_layout("b", "x", "y"), in_layout("b", "y", "z"))
    net_parallel = BatchMatMulCell(in_strategy, (128, 96, 128))
    net_parallel.batchmatmul.add_prim_attr("enable_nd_tp", True)
    parallel_dataset = FakeData(size=256, batch_size=16, image_size=(128, 96), num_classes=128,
                                use_parallel=True)
    parallel_model = modeltrainbase.create_train_model(net_parallel, loss=None)
    with pytest.raises(RuntimeError):
        modeltrainbase.load_newest_ckpt_from_model_train(parallel_model, epoch=1,
                                                         dataset=parallel_dataset,
                                                         dataset_sink_mode=False,
                                                         ckpt_path=f"./parallel_batchmatmul_high_dim/rank_{rank_id}",
                                                         ckpt_prefix="parallel")
