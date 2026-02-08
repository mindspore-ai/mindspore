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
"""Test pipeline interleaving schedulers (GPipe and 1F1B) for distributed training.

This module validates pipeline parallelism with different interleaving scheduling strategies
in MindSpore. It compares two popular pipeline schedulers: GPipe (sequential with interleaving)
and 1F1B (One Forward One Backward with multiple in-flight microbatches), both designed to
hide communication latency and improve GPU utilization during distributed training.

Testing Workflow:

GPipe Scheduler Test:
1. Initialize 8 devices with 2-stage pipeline parallelism
2. Configure semi-automatic parallel mode with pipeline_interleave=True
3. Create LeNet-style network with conv1-conv2-fc1-fc2-fc3 architecture
4. Apply sharding strategy to conv1 layer across 4 devices: ((4, 1, 1, 1), (1, 1, 1, 1))
5. Assign pipeline stages: Stage 0 (conv1, fc1), Stage 1 (conv2, fc2, fc3)
6. Create PipelineCell with 2 microbatches for GPipe scheduling
7. Train on fake dataset and save checkpoints per rank

1F1B Scheduler Test:
1. Initialize 8 devices with 2-stage pipeline parallelism
2. Configure semi-automatic parallel mode with pipeline_scheduler="1f1b"
3. Create identical LeNet network with same sharding strategy
4. Assign same pipeline stages for consistency
5. Create PipelineCell with 4 microbatches for 1F1B scheduling
6. Train on fake dataset and save checkpoints per rank

Key Differences Between Schedulers:
- GPipe: Processes all microbatches sequentially (forward all, backward all)
  * Lower memory overhead
  * Higher pipeline bubble
  * Better for memory-constrained scenarios

- 1F1B: Alternates forward and backward passes (1 forward, 1 backward per microbatch)
  * Higher memory requirement for in-flight microbatches
  * Reduced pipeline bubble with better GPU utilization
  * Improved throughput for typical workloads

Key Components:
- Net: LeNet-style CNN with 2 conv layers and 3 dense layers
- PipelineCell: Wrapper for pipeline parallelism execution
- FakeData: Synthetic dataset for testing
- modeltrainbase: Training utilities for distributed execution

Test Configuration:
- Device count: 8 (4 devices per pipeline stage)
- Pipeline stages: 2
- Input shape: (batch_size, 1, 32, 32)
- Conv1 sharding: Distributed across 4 devices in batch dimension
"""
import mindspore as ms
import numpy as np
from mindspore import nn, context
from mindspore.communication.management import init
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.common import lazy_inline
from mindspore.communication.management import get_rank
from .utils.modeltrain_base import modeltrainbase
from .model_parallel import FakeData


def setup_function():
    """Initialize the test environment with context and communication management.
    
    This setup function is automatically called before running the tests in this module.
    It sets MindSpore to GRAPH_MODE (0), enables graph saving for debugging, and 
    initializes the distributed communication framework.
    """
    ms.set_context(mode=0)
    ms.set_context(save_graphs=True, save_graphs_path="./pipeline_interleave/ir")
    init()


class Net(nn.Cell):
    """LeNet-style convolutional neural network for pipeline interleaving testing.
    
    This network consists of two convolutional layers followed by three fully-connected layers,
    designed to demonstrate pipeline parallelism with interleaving schedulers (GPipe and 1F1B).
    The network architecture allows for effective stage-wise partitioning across multiple devices.
    
    Attributes:
        conv1: First convolutional layer (1 -> 6 channels, kernel_size=5)
        conv2: Second convolutional layer (6 -> 16 channels, kernel_size=5)
        fc1: First fully-connected layer (400 -> 120 units)
        fc2: Second fully-connected layer (120 -> 84 units)
        fc3: Output layer (84 -> 10 units)
        relu: ReLU activation function
        max_pool2d: Max pooling layer (kernel_size=2, stride=2)
        flatten: Flattening layer
    """
    @lazy_inline
    def __init__(self):
        """Initialize the Net network layers.
        
        Sets up all convolutional, fully-connected, and activation layers.
        The @lazy_inline decorator is used to optimize the network during compilation.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, pad_mode='valid',
                               weight_init='normal')
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, pad_mode='valid',
                               weight_init='normal')
        self.fc1 = nn.Dense(in_channels=16 * 5 * 5, out_channels=120, weight_init='normal',
                            bias_init='zeros')
        self.fc2 = nn.Dense(in_channels=120, out_channels=84, weight_init='normal',
                            bias_init='zeros')
        self.fc3 = nn.Dense(in_channels=84, out_channels=10, weight_init='normal',
                            bias_init='zeros')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.flatten = nn.Flatten()

    def construct(self, x, label):
        """Forward pass of the network.
        
        Applies convolutional layers, activation functions, pooling, and fully-connected layers
        sequentially to transform input tensor to output predictions.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 32, 32).
            label: Label tensor (unused, passed for compatibility with training pipeline).
            
        Returns:
            Output tensor of shape (batch_size, 10) containing class predictions.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def test_pipeline_interleave_001():
    """Test pipeline interleaving with GPipe scheduler.
    
    This test validates pipeline parallelism with interleaving enabled using the GPipe scheduling
    strategy. The test covers:
    - 8 devices with 2 pipeline stages
    - Semi-automatic parallel mode
    - GPipe scheduler for pipeline interleaving
    - Network sharding strategy with conv1 layer distributed across 4 devices
    - Pipeline stage assignment for conv1 and fc1 to stage 0, others to stage 1
    - Training with fake dataset and checkpoint saving per rank
    
    The GPipe scheduler processes all microbatches sequentially but uses interleaving to hide
    communication latency during the backward pass.
    """
    # Seed the random number generator for reproducibility
    np.random.seed(100)
    rank_id = get_rank()
    context.reset_auto_parallel_context()

    # Configure pipeline parallelism with GPipe interleaving scheduler
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel",
                                      pipeline_stages=2,
                                      pipeline_config={"pipeline_interleave": True,
                                                       "pipeline_scheduler": "gpipe"})

    # Create network and apply sharding strategy
    net1 = Net()
    net1.conv1.conv2d.shard(((4, 1, 1, 1), (1, 1, 1, 1)))

    # Assign layers to pipeline stages
    # Stage 0: conv1, fc1 (initial processing layers)
    # Stage 1: conv2, fc2, fc3 (final processing layers)
    net1.conv1.pipeline_stage = 0
    net1.conv2.pipeline_stage = 1
    net1.fc1.pipeline_stage = 0
    net1.fc2.pipeline_stage = 1
    net1.fc3.pipeline_stage = 1

    # Create pipeline cell with 2 stages
    pipeline_net = PipelineCell(net1, 2)

    # Create fake dataset for testing
    parallel_dataset = FakeData(size=256, batch_size=256, image_size=(1, 32, 32))

    # Create training model and train with checkpoint saving
    parallel_model = modeltrainbase.create_train_model(pipeline_net, loss=None)
    modeltrainbase.load_newest_ckpt_from_model_train(parallel_model, epoch=1,
                                                     dataset=parallel_dataset,
                                                     dataset_sink_mode=False,
                                                     integrated_save=False,
                                                     ckpt_path=f"./pipeline_interleave/rk_{rank_id}_ckpt",
                                                     ckpt_prefix="parallel")


def test_pipeline_interleave_004():
    """Test pipeline interleaving with 1F1B (One Forward One Backward) scheduler.
    
    This test validates pipeline parallelism with interleaving enabled using the 1F1B scheduling
    strategy. The test covers:
    - 8 devices with 2 pipeline stages
    - Semi-automatic parallel mode
    - 1F1B scheduler for pipeline interleaving
    - Network sharding strategy with conv1 layer distributed across 4 devices
    - Pipeline stage assignment for conv1 and fc1 to stage 0, others to stage 1
    - PipelineCell with 4 microbatches for interleaving
    - Training with fake dataset and checkpoint saving per rank
    
    The 1F1B scheduler allows multiple microbatches to be in-flight simultaneously,
    enabling better GPU utilization by overlapping computation and communication.
    """
    # Seed the random number generator for reproducibility
    np.random.seed(100)
    rank_id = get_rank()
    context.reset_auto_parallel_context()

    # Configure pipeline parallelism with 1F1B interleaving scheduler
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel",
                                      pipeline_stages=2,
                                      pipeline_config={"pipeline_interleave": True,
                                                       "pipeline_scheduler": "1f1b"})

    # Create network and apply sharding strategy
    net1 = Net()
    net1.conv1.conv2d.shard(((4, 1, 1, 1), (1, 1, 1, 1)))

    # Assign layers to pipeline stages
    # Stage 0: conv1, fc1 (initial processing layers)
    # Stage 1: conv2, fc2, fc3 (final processing layers)
    net1.conv1.pipeline_stage = 0
    net1.conv2.pipeline_stage = 1
    net1.fc1.pipeline_stage = 0
    net1.fc2.pipeline_stage = 1
    net1.fc3.pipeline_stage = 1

    # Create pipeline cell with 4 microbatches for 1F1B scheduling
    pipeline_net = PipelineCell(net1, 4)

    # Create fake dataset for testing
    parallel_dataset = FakeData(size=256, batch_size=256, image_size=(1, 32, 32))

    # Create training model and train with checkpoint saving
    parallel_model = modeltrainbase.create_train_model(pipeline_net, loss=None)
    modeltrainbase.load_newest_ckpt_from_model_train(parallel_model, epoch=1,
                                                     dataset=parallel_dataset,
                                                     dataset_sink_mode=False,
                                                     integrated_save=False,
                                                     ckpt_path=f"./pipeline_interleave/rk_{rank_id}_ckpt",
                                                     ckpt_prefix="parallel")
