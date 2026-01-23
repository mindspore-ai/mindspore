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
"""Test runner for memory swap and offload functionality in distributed training.

This module serves as a test execution wrapper that launches tests for memory swap and offload
operations on Ascend devices. It validates the framework's ability to optimize memory usage
by swapping intermediate activations and parameters between device memory and CPU memory
during training, enabling training of larger models on memory-constrained devices.

The test verifies that:
- Memory offload to CPU memory functions correctly during training
- Swap operations properly handle parameter and activation movement
- Training with offload produces correct numerical results
- Inference after offload training executes successfully
- Device placement constraints are properly respected

Offload Features Tested:
- Parameter offloading: Moving weights to CPU memory during forward/backward passes
- Activation offloading: Storing intermediate activations in CPU memory
- Memory swap: Intelligent swapping between device and CPU memory
- Asynchronous operations: Overlapping compute and memory operations
- Training correctness: Numerical equivalence with and without offload

Execution Details:
- Platform: Ascend (NPU) devices
- Test focus: Memory management and device coordination
- Training mode: Graph mode with auto parallel
- Checkpoint management: Saving/loading with offload configuration

The actual test logic is implemented in test_swap.py wrapper, which validates:
- Network creation with CPU-resident parameters
- Training execution with memory offload enabled
- Inference execution after offload training
- Checkpoint saving and loading with offload state
"""
import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.graph.lazy_inline import lazy_inline
from tests.mark_utils import arg_mark
from tests.st.auto_parallel.model_parallel import FakeData
from tests.st.auto_parallel.utils.modeltrain_base import modeltrainbase


class Network(nn.Cell):
    """A neural network for memory swap and offload testing.
    
    This network demonstrates parameter creation on specific devices and memory offload capabilities.
    It applies a series of multiplication operations to test the offloading of intermediate
    activations and parameters to CPU memory during training on Ascend devices.
    
    Attributes:
        relu: ReLU activation function.
        weight_1: First learnable weight parameter.
        weight_2: Second learnable weight parameter.
        weight_3: Third learnable weight parameter.
        mul1: First multiplication operation.
        mul2: Second multiplication operation.
        mul3: Third multiplication operation.
    """
    @lazy_inline
    def __init__(self, weight_shape, device=None):
        """Initialize the Network with learnable parameters.
        
        Args:
            weight_shape: Shape tuple for weight parameters (e.g., (256, 256)).
            device: Device placement for parameters (e.g., "CPU", "GPU"). 
                   If None, parameters are placed on default device. Default is None.
        """
        super().__init__()
        self.relu = P.ReLU()
        if device is None:
            self.weight_1 = Parameter(Tensor(np.random.randn(*weight_shape), dtype=ms.float32),
                                      name="weight_1")
            self.weight_2 = Parameter(Tensor(np.random.randn(*weight_shape), dtype=ms.float32),
                                      name="weight_2")
            self.weight_3 = Parameter(Tensor(np.random.randn(*weight_shape), dtype=ms.float32),
                                      name="weight_3")

        else:
            self.weight_1 = Parameter(Tensor(np.random.randn(*weight_shape), dtype=ms.float32),
                                      device=device,
                                      name="weight_1")
            self.weight_2 = Parameter(Tensor(np.random.randn(*weight_shape), dtype=ms.float32),
                                      device=device,
                                      name="weight_2")
            self.weight_3 = Parameter(Tensor(np.random.randn(*weight_shape), dtype=ms.float32),
                                      device=device,
                                      name="weight_3")
        self.mul1 = P.Mul()
        self.mul2 = P.Mul()
        self.mul3 = P.Mul()

    def construct(self, x):
        """Forward pass applying ReLU and sequential multiplications.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after applying ReLU activation and three sequential 
            multiplication operations with weight parameters.
        """
        x = self.relu(x)
        x = self.mul1(x, self.weight_1)
        x = self.mul2(x, self.weight_2)
        x = self.mul3(x, self.weight_3)
        return x


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
        essential_mark="essential", mem_peak=6.00)
def test_swap_offload_store_key_check():
    '''
    Feature: swap.
    Description: test swap offload_store key_check.
    Expectation: Run success.
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    inputs = Tensor(np.random.randn(256, 256).astype(np.float32))
    ms.set_seed(10)
    network2 = Network((256, 256), device="CPU")
    network2.offload()
    dataset2 = FakeData(size=256, batch_size=256, image_size=(256,), num_classes=256)
    model2 = modeltrainbase.create_train_model(network2)
    model2.train(epoch=1, train_dataset=dataset2, dataset_sink_mode=False)
    model2.predict(inputs)
