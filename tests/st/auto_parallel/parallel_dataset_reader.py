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
"""Test pipeline parallelism with dataset sharding strategy and distributed checkpoint management.

This module validates the end-to-end workflow of distributed training with pipeline parallelism,
dataset-level sharding strategies, and checkpoint synchronization across multiple ranks. It tests
the MindSpore framework's ability to handle complex distributed training scenarios with 2-stage
pipeline parallelism, semi-automatic parallel mode, and dataset sharding optimizations.

Testing Workflow:
1. Initialize distributed environment with 8 devices and semi-automatic parallel mode
2. Configure pipeline parallelism with 2 stages, dataset sharding strategies
3. Define network with two BiasAdd stages connected through multiplication layer
4. Phase 1: Train with pipeline1 configuration and save checkpoints to rk_*_ckpt directories
5. Phase 2: Configure dataset speed-up optimization and train with pipeline2 configuration
6. Phase 3: Load distributed checkpoints and execute inference on both configurations
7. Verify dataset optimization (Broadcast operations) in compiled IR graphs
8. Validate inference consistency across different pipeline configurations

Validation Points:
- Dataset sharding strategy is correctly applied across ranks ((1, 1, 1, 2))
- Broadcast operations are present in dataset optimization IR graphs
- Distributed checkpoints can be properly loaded from pipeline-specific directories
- Inference results are numerically consistent for stage-1 ranks (>= rank 4)
- Different pipeline configurations produce equivalent predictions

Key Components:
- WithLossCell: Network wrapper combining backbone with loss function
- ParallelBiasAddNet: BiasAdd + multiplication operations with distributed sharding
- ParallelMulNet: Element-wise multiplication with optional sharding strategy
- PipeDatasetStrategy: Multi-stage pipeline network with two BiasAdd stages
- mindspore_pipeline_predict: Distributed inference function with checkpoint loading
- test_parallel_dataset_reader_dataset_strategy_pipeline_002: Main test function

Pipeline Configuration:
- Device count: 8 (4 devices per stage for 2-stage pipeline)
- Pipeline stages: 2 (stage 0: net1, stage 1: mul + net2)
- Dataset sharding: Full 4D tensor sharding with (1, 1, 1, 2) strategy
- Network shape: (16, 3, 16, 16) for input, same for parameters
"""
import os
import subprocess
import numpy as np
import mindspore as ms
from dataset.animal import create_splited_data_dataset
from mindspore import nn, Parameter, Tensor, context
from mindspore.nn import L1Loss
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from mindspore.train.model import Model
import mindspore.ops.operations as P
from mindspore.communication.management import init
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.common import lazy_inline
from mindspore import load_distributed_checkpoint
from mindspore.context import ParallelMode
from tests.st.auto_parallel.utils.modeltrain_base import modeltrainbase
from tests.st.auto_parallel.utils.utils import allclose_nparray


def setup_function():
    """Initialize the test environment with context and communication management.
    
    This setup function is automatically called before running the tests in this module.
    It sets MindSpore to GRAPH_MODE (0), enables graph saving for debugging, and 
    initializes the distributed communication framework.
    """
    ms.set_context(mode=0)
    ms.set_context(save_graphs=True, save_graphs_path="./ir")
    init()


class WithLossCell(nn.Cell):
    """A wrapper cell that combines a backbone network with a loss function.
    
    This class encapsulates a neural network backbone and a loss function,
    automatically computing the loss on the model output during the forward pass.
    It is commonly used in training pipelines to simplify the training loop.
    """
    @lazy_inline
    def __init__(self, backbone, loss_fn):
        """Initialize the WithLossCell.
        
        Args:
            backbone: The neural network backbone model to be wrapped.
            loss_fn: The loss function to compute the loss between model output and labels.
        """
        super().__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._get_attr_from_cell(backbone)

    def construct(self, data, label):
        """Forward pass computing backbone output and loss.
        
        Args:
            data: Input data tensor to the backbone network.
            label: Ground truth labels for computing the loss.
            
        Returns:
            Loss value computed between backbone output and labels.
        """
        out = self._backbone(data)
        return self._loss_fn(out, label)


class ParallelBiasAddNet(nn.Cell):
    """A neural network cell combining multiplication and bias addition operations with sharding strategies.
    
    This class implements a simple network that performs element-wise multiplication with a learnable
    weight parameter, followed by bias addition. Both operations can be sharded across multiple
    devices according to specified strategies for distributed training.
    
    Attributes:
        mul_weight: Learnable weight parameter for the multiplication operation.
        bias: Learnable bias parameter for the bias addition operation.
        mul: Element-wise multiplication operation.
        bias_add: Bias addition operation.
    """
    def __init__(self, mul_size, bias_size, strategy=None, strategy2=None):
        """Initialize the ParallelBiasAddNet.
        
        Args:
            mul_size: Shape tuple for the multiplication weight parameter.
            bias_size: Shape tuple for the bias parameter.
            strategy: Sharding strategy for the bias_add operation. Default is None.
            strategy2: Sharding strategy for the mul operation. Default is None.
        """
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float32)
        bias_np = np.full(bias_size, 0.1, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.bias = Parameter(Tensor(bias_np), name="bias")
        self.mul = P.Mul()
        self.bias_add = P.BiasAdd()

        if strategy is not None:
            self.mul.shard(strategy2)
            self.bias_add.shard(strategy)

    def construct(self, inputs):
        """Forward pass applying multiplication and bias addition.
        
        Args:
            inputs: Input tensor.
            
        Returns:
            Output tensor after multiplication and bias addition operations.
        """
        x = self.mul(inputs, self.mul_weight)
        x = self.bias_add(x, self.bias)
        return x


class ParallelMulNet(nn.Cell):
    """A neural network cell that performs element-wise multiplication with sharding strategy.
    
    This is a lightweight operation cell that performs element-wise multiplication and supports
    distributed sharding across multiple devices.
    """
    def __init__(self, strategy=None):
        """Initialize the ParallelMulNet.
        
        Args:
            strategy: Optional sharding strategy for the multiplication operation. Default is None.
        """
        super().__init__()
        self.mul = P.Mul()
        self.mul.shard(strategy)

    def construct(self, x, y):
        """Forward pass performing element-wise multiplication.
        
        Args:
            x: First input tensor.
            y: Second input tensor.
            
        Returns:
            Result of element-wise multiplication between x and y.
        """
        return self.mul(x, y)


class PipeDatasetStrategy(nn.Cell):
    """A multi-stage neural network combining bias-add and multiplication operations.
    
    This network consists of two BiasAddNet stages connected through a multiplication layer,
    designed to demonstrate pipeline parallelism and dataset sharding strategies in distributed training.
    
    Attributes:
        net1: First parallel bias-add network (stage 0).
        net2: Second parallel bias-add network (stage 1).
        mul: Parallel multiplication layer connecting both stages.
    """
    def __init__(self, mul_size, bias_size, strategy, strategy2, mul_strategy):
        """Initialize the PipeDatasetStrategy network.
        
        Args:
            mul_size: Tuple of size tuples for the two networks' multiplication parameters.
            bias_size: Tuple of size tuples for the two networks' bias parameters.
            strategy: Tuple of sharding strategies for bias_add operations in both networks.
            strategy2: Tuple of sharding strategies for mul operations in both networks.
            mul_strategy: Sharding strategy for the connecting multiplication layer.
        """
        super().__init__()
        self.net1 = ParallelBiasAddNet(mul_size[0], bias_size[0], strategy[0], strategy2[0])
        self.net2 = ParallelBiasAddNet(mul_size[1], bias_size[1], strategy[1], strategy2[1])
        self.mul = ParallelMulNet(mul_strategy)

    def construct(self, x):
        """Forward pass through the multi-stage pipeline.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after processing through net1, mul layer, and net2 sequentially.
        """
        out = self.net1(x)
        out = self.mul(x, out)
        out = self.net2(out)
        return out


def find_files(file, para):
    """Search for a parameter string in a file and count occurrences.
    
    Uses grep command to search for occurrences of a parameter string in a file.
    This is typically used to verify that certain operations or optimizations
    have been applied during graph compilation.
    
    Args:
        file: Path to the file to search in.
        para: Parameter or pattern string to search for.
        
    Returns:
        String representation of the count of matching lines.
    """
    output = subprocess.check_output(["grep '%s' %s|wc -l" % (para, file)], shell=True)
    out = str(output, 'utf-8').strip()
    return out


def mindspore_pipeline_predict(net, predict_data, dataset_strategy="full_batch",
                               parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stage=2,
                               strategy_ckpt_load_file="./strategy_stage1.ckpt", device_num=8,
                               search_mode="sharding_propagation", global_rank_id=None,
                               ckpt_file="parallel-1_1.ckpt", enable_parallel_optimizer=False,
                               skip_backend_compile=False):
    """Execute model inference with pipeline parallelism and distributed checkpoints.
    
    This function performs distributed inference on a model using pipeline parallelism.
    It loads the model's distributed checkpoint files from multiple ranks, constructs
    the predict strategy layout, and executes inference on the provided data.
    
    Args:
        net: The neural network model for inference.
        predict_data: Input tensor for prediction.
        dataset_strategy: Dataset sharding strategy. Default is "full_batch".
        parallel_mode: Parallel execution mode. Default is SEMI_AUTO_PARALLEL.
        pipeline_stage: Number of pipeline stages. Default is 2.
        strategy_ckpt_load_file: Path to the strategy checkpoint file to load. Default is "./strategy_stage1.ckpt".
        device_num: Total number of devices. Default is 8.
        search_mode: Auto parallel search mode. Default is "sharding_propagation".
        global_rank_id: Current rank ID in distributed environment.
        ckpt_file: Checkpoint file name to load for each rank. Default is "parallel-1_1.ckpt".
        enable_parallel_optimizer: Whether to enable parallel optimizer. Default is False.
        skip_backend_compile: Whether to skip backend compilation. Default is False.
        
    Returns:
        Inference result as a tensor.
    """
    context.reset_auto_parallel_context()
    num = int(device_num / pipeline_stage)
    if "parallel" in ckpt_file:
        if pipeline_stage == 2:
            if global_rank_id < num:
                pipe_ckpt_file = [f"./rk_{i}_ckpt/{ckpt_file}" for i in range(num)]
            else:
                pipe_ckpt_file = [f"./rk_{i}_ckpt/{ckpt_file}" for i in range(num, num * 2)]
        else:
            if global_rank_id < num:
                pipe_ckpt_file = [f"./rk_{i}_ckpt/{ckpt_file}" for i in range(num)]
            elif num <= global_rank_id < (num * 2):
                pipe_ckpt_file = [f"./rk_{i}_ckpt/{ckpt_file}" for i in range(num, num * 2)]
            elif num * 2 <= global_rank_id < (num * 3):
                pipe_ckpt_file = [f"./rk_{i}_ckpt/{ckpt_file}" for i in range(num * 2, num * 3)]
            else:
                pipe_ckpt_file = [f"./rk_{i}_ckpt/{ckpt_file}" for i in range(num * 3, num * 4)]
    else:
        if pipeline_stage == 2:
            if global_rank_id < num:
                pipe_ckpt_file = [f"./rank_{i}_ckpt/{ckpt_file}" for i in range(num)]
            else:
                pipe_ckpt_file = [f"./rank_{i}_ckpt/{ckpt_file}" for i in range(num, num * 2)]
        else:
            if global_rank_id < num:
                pipe_ckpt_file = [f"./rank_{i}_ckpt/{ckpt_file}" for i in range(num)]
            elif num <= global_rank_id < (num * 2):
                pipe_ckpt_file = [f"./rank_{i}_ckpt/{ckpt_file}" for i in range(num, num * 2)]
            elif num * 2 <= global_rank_id < (num * 3):
                pipe_ckpt_file = [f"./rank_{i}_ckpt/{ckpt_file}" for i in range(num * 2, num * 3)]
            else:
                pipe_ckpt_file = [f"./rank_{i}_ckpt/{ckpt_file}" for i in range(num * 3, num * 4)]

    context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                      enable_parallel_optimizer=enable_parallel_optimizer,
                                      dataset_strategy=dataset_strategy,
                                      pipeline_stages=pipeline_stage, search_mode=search_mode,
                                      strategy_ckpt_config={"load_file": strategy_ckpt_load_file},
                                      group_ckpt_save_file=f"./group_config_{global_rank_id}.pb")
    if enable_parallel_optimizer:
        context.set_auto_parallel_context(parallel_optimizer_config={'parallel_optimizer_threshold': 0})
    model = Model(net)
    predict_strategy = model.infer_predict_layout(predict_data, skip_backend_compile=skip_backend_compile)
    load_distributed_checkpoint(net, pipe_ckpt_file, predict_strategy)
    infer = model.predict(predict_data)
    context.reset_auto_parallel_context()
    return infer


def test_parallel_dataset_reader_dataset_strategy_pipeline_002():
    """Test pipeline parallelism with dataset sharding strategy and checkpoint loading/saving.
    
    This comprehensive test validates:
    1. Pipeline parallelism with 2 stages across 8 devices
    2. Dataset sharding strategy application across ranks
    3. Training with checkpoint saving in two different configurations
    4. Loading and executing inference with distributed checkpoints
    5. Verifying that different dataset strategies produce consistent results
    
    The test performs three main phases:
    Phase 1: Train and save checkpoints with pipeline1 configuration
    Phase 2: Train and save checkpoints with pipeline2 configuration
    Phase 3: Load both checkpoint configurations and verify inference consistency
    
    Test validates that:
    - Graph optimizations (e.g., Broadcast operations) are correctly applied
    - Different dataset strategies produce numerically consistent results
    - Distributed checkpoints can be properly loaded for inference
    """
    # Initialize distributed communication
    D.init()
    os.environ['MS_DEV_SAVE_GRAPHS'] = "3"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = "./ir"
    rank_id = get_rank()

    # ===== Phase 1: Train with pipeline1 configuration =====
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel",
                                     pipeline_stages=2,
                                     enable_parallel_optimizer=True,
                                     dataset_strategy=((1, 1, 1, 2), (1, 1, 1, 2)),
                                     parallel_optimizer_config={"parallel_optimizer_threshold": 0},
                                     strategy_ckpt_config={
                                         "save_file": f"./pipeline1_strategy_stage{rank_id}.json"})
    parallel_net = PipeDatasetStrategy(mul_size=((16, 3, 16, 16), (16, 3, 16, 16)),
                                       bias_size=((3,), (3,)),
                                       strategy=(((1, 1, 2, 1), (1,)), ((1, 1, 2, 1), (1,))),
                                       strategy2=(
                                           ((1, 1, 2, 1), (1, 1, 2, 1)),
                                           ((1, 1, 2, 1), (1, 1, 2, 1))),
                                       mul_strategy=((1, 1, 1, 2), (1, 1, 1, 2)))
    parallel_net.net1.pipeline_stage = 0
    parallel_net.mul.pipeline_stage = 1
    parallel_net.net2.pipeline_stage = 1
    parallel_data = create_splited_data_dataset(img_h=1, img_w=2, mask_h=1, mask_w=2, batch_size=32,
                                                resize_height=16, resize_width=16)
    loss = L1Loss()
    loss_net_parallel = WithLossCell(parallel_net, loss)
    auto_parallel_net = PipelineCell(loss_net_parallel, 2)
    parallel_model = modeltrainbase.create_train_model(auto_parallel_net, loss=None)
    _ = modeltrainbase.load_newest_ckpt_from_model_train(parallel_model, epoch=1,
                                                         dataset=parallel_data,
                                                         dataset_sink_mode=True, sink_size=1,
                                                         integrated_save=False,
                                                         ckpt_path=f"./rk_{rank_id}_ckpt",
                                                         ckpt_prefix="parallel")

    # ===== Phase 2: Train with pipeline2 configuration =====
    context.reset_auto_parallel_context()
    context.set_context(ascend_config={
                            "parallel_speed_up_json_path": "./dataset/parallel_speed_up.json"})
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel",
                                     enable_parallel_optimizer=True,
                                     parallel_optimizer_config={"parallel_optimizer_threshold": 0},
                                     pipeline_stages=2,
                                     dataset_strategy=((1, 1, 1, 2), (1, 1, 1, 2)),
                                     strategy_ckpt_config={
                                         "save_file": f"./pipeline2_strategy_stage{rank_id}.json"})
    parallel_net = PipeDatasetStrategy(mul_size=((16, 3, 16, 16), (16, 3, 16, 16)),
                                       bias_size=((3,), (3,)),
                                       strategy=(((1, 1, 2, 1), (1,)), ((1, 1, 2, 1), (1,))),
                                       strategy2=(
                                           ((1, 1, 2, 1), (1, 1, 2, 1)),
                                           ((1, 1, 2, 1), (1, 1, 2, 1))),
                                       mul_strategy=((1, 1, 1, 2), (1, 1, 1, 2)))
    parallel_net.net1.pipeline_stage = 0
    parallel_net.mul.pipeline_stage = 1
    parallel_net.net2.pipeline_stage = 1
    parallel_data = create_splited_data_dataset(img_h=1, img_w=2, mask_h=1, mask_w=2, batch_size=32,
                                                resize_height=16, resize_width=16)
    loss = L1Loss()
    loss_net_parallel = WithLossCell(parallel_net, loss)
    auto_parallel_net = PipelineCell(loss_net_parallel, 2)
    parallel_model = modeltrainbase.create_train_model(auto_parallel_net, loss=None)
    _ = modeltrainbase.load_newest_ckpt_from_model_train(parallel_model, epoch=1,
                                                         dataset=parallel_data,
                                                         dataset_sink_mode=True, sink_size=1,
                                                         integrated_save=False,
                                                         ckpt_path=f"./rank_{rank_id}_ckpt",
                                                         ckpt_prefix="pipeline")

    # Verify that dataset optimization (Broadcast) is applied in the IR graphs
    out = find_files(f"./ir/rank_{rank_id}/*_dataset_repeat_opt_*.ir", "Broadcast")
    assert int(out) >= 1

    # ===== Phase 3: Inference with loaded checkpoints =====
    context.reset_auto_parallel_context()
    net = PipeDatasetStrategy(mul_size=((32, 3, 16, 16), (32, 3, 16, 16)),
                              bias_size=((3,), (3,)),
                              strategy=(((1, 1, 2, 1), (1,)), ((1, 1, 2, 1), (1,))),
                              strategy2=(
                                  ((1, 1, 2, 1), (1, 1, 2, 1)), ((1, 1, 2, 1), (1, 1, 2, 1))),
                              mul_strategy=((1, 1, 1, 2), (1, 1, 1, 2)))
    net.net1.pipeline_stage = 0
    net.mul.pipeline_stage = 1
    net.net2.pipeline_stage = 1
    predict_data = Tensor(np.random.randn(32, 3, 16, 16).astype(np.float32))
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel",
                                     enable_parallel_optimizer=True,
                                     parallel_optimizer_config={"parallel_optimizer_threshold": 0},
                                     pipeline_stages=2)

    # Load and run inference with pipeline1 configuration
    infer1 = mindspore_pipeline_predict(net, predict_data, pipeline_stage=2,
                                        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                        global_rank_id=rank_id, ckpt_file="parallel-1_1.ckpt",
                                        strategy_ckpt_load_file=f"./pipeline1_strategy_stage{rank_id}.json")

    # Load and run inference with pipeline2 configuration
    infer2 = mindspore_pipeline_predict(net, predict_data, pipeline_stage=2,
                                        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                        global_rank_id=rank_id, ckpt_file="pipeline-1_1.ckpt",
                                        strategy_ckpt_load_file=f"./pipeline2_strategy_stage{rank_id}.json")

    # Verify inference consistency for ranks in stage 1 (ranks >= 4)
    if rank_id >= 4:
        allclose_nparray(infer1.asnumpy(), infer2.asnumpy(), 0.005, 0.005)
