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
"""Test ZeRO optimizer level 3 with data parallelism 4 and pipeline parallelism 2.

Tests distributed training with DP4+PP2+ZeRO3 configuration including checkpoint saving.
"""
import mindspore as ms
import numpy as np
from mindspore import nn, context
from mindspore.nn import PipelineCell
from mindspore.common import lazy_inline
import mindspore.communication.management as D
from mindspore.communication.management import init, get_rank
from tests.st.auto_parallel.utils.modeltrain_base import modeltrainbase
from tests.st.auto_parallel.model_parallel import FakeData


def setup_function():
    """Initialize the test environment with context and communication management.
    
    This setup function is automatically called before running the tests in this module.
    It sets MindSpore to GRAPH_MODE (0), enables graph saving for debugging, and 
    initializes the distributed communication framework.
    """
    ms.set_context(mode=0)
    ms.set_context(save_graphs=True, save_graphs_path="./parallel_zero_optimizer_level/ir")
    init()


class Net1(nn.Cell):
    """LeNet-style convolutional neural network for parallel training and pipeline stages testing.
    
    This network consists of two convolutional layers followed by three fully-connected layers,
    designed to demonstrate distributed training with pipeline parallelism and ZeRO optimization.
    
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
        """Initialize the Net1 network layers.
        
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

    def construct(self, x, _):
        """Forward pass of the network.
        
        Applies convolutional layers, activation functions, pooling, and fully-connected layers
        sequentially to transform input tensor to output predictions.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 32, 32).
            _: Unused parameter, typically used for auxiliary loss in parallel training.
            
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


def test_parallel_optimizer_dp4vpp2_zero3():
    """Test parallel optimizer with Data Parallelism 4 and Vertext Pipeline Parallelism 2 using ZeRO level 3.
    
    This test validates the integration of:
    - Data parallelism across 4 devices
    - Pipeline parallelism across 2 stages
    - ZeRO optimization level 3 for memory efficiency
    - Pipeline interleaving with GPipe scheduler
    
    The test creates a Net1 network with specific sharding and pipeline stage assignments,
    then trains it on fake data with checkpoint saving.
    
    Raises:
        Various exceptions if the distributed training setup or model training fails.
    """
    D.init()
    # Seed the random number generator for reproducibility
    np.random.seed(100)
    rank_id = get_rank()
    context.reset_auto_parallel_context()

    # Configure ZeRO optimization level 3 with weight sharding
    optim_cfg = {"optimizer_level": "level3", \
                 "optimizer_weight_shard_size": 2, \
                 "parallel_optimizer_threshold": 0}

    # Set up automatic parallel context with 8 devices total, semi-auto parallel mode
    # Configure 2 pipeline stages with pipeline interleaving enabled
    context.set_auto_parallel_context(device_num=8, parallel_mode="semi_auto_parallel", \
                                     enable_parallel_optimizer=True, parallel_optimizer_config=optim_cfg, \
                                     pipeline_stages=2,
                                     pipeline_config={"pipeline_interleave": True,
                                                      "pipeline_scheduler": "gpipe"})

    # Create network instance and configure sharding strategy
    net1 = Net1()
    net1.conv1.conv2d.shard(((4, 1, 1, 1), (1, 1, 1, 1)))

    # Assign layers to pipeline stages for distributed computation
    # Stage 0: conv1, fc1 (first layers for initial processing)
    # Stage 1: conv2, fc2, fc3 (remaining layers for final processing)
    net1.conv1.pipeline_stage = 0
    net1.conv2.pipeline_stage = 1
    net1.fc1.pipeline_stage = 0
    net1.fc2.pipeline_stage = 1
    net1.fc3.pipeline_stage = 1

    # Create pipeline cell with 2 stages for executing distributed pipeline parallelism
    pipeline_net = PipelineCell(net1, 2)

    # Create fake dataset for testing (256 samples, batch size 256, image size 1x32x32)
    parallel_dataset = FakeData(size=256, batch_size=256, image_size=(1, 32, 32))

    # Create training model with pipeline network
    parallel_model = modeltrainbase.create_train_model(pipeline_net, loss=None)

    # Load and train the model, saving checkpoints per rank
    modeltrainbase.load_newest_ckpt_from_model_train(parallel_model, epoch=1,
                                                     dataset=parallel_dataset,
                                                     dataset_sink_mode=False,
                                                     integrated_save=False,
                                                     ckpt_path=f"./parallel_zero_optimizer_level/rk_{rank_id}_ckpt",
                                                     ckpt_prefix="parallel")
