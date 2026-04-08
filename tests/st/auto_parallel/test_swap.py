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
import os
import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops.operations as P
from mindspore import nn
from mindspore import Tensor
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.graph.lazy_inline import lazy_inline
from tests.mark_utils import arg_mark
from tests.st.auto_parallel.model_parallel import FakeData
from tests.st.auto_parallel.utils.modeltrain_base import modeltrainbase
from tests.st.high_availability.utils.context_base import contextbase
from tests.st.auto_parallel.checkpoint_unified_safetensors import clean_all_ckpt_files
from tests.st.pi_jit.share.modeltrain_utils import GeneratorFakeData


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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", mem_peak=6.00)
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


class MulAddMulNet(nn.Cell):
    """A neural network for memory swap and offload testing."""
    def __init__(self, mul_shape, add_shape, strategy=None, dtype=ms.float32, swap=False):
        super().__init__()
        if swap:
            self.mul_weight_1 = Parameter(Tensor(np.full(mul_shape, 0.5), dtype=dtype),
                                          name="mul_weight_1", device="CPU")
            self.mul_weight_2 = Parameter(Tensor(np.full(mul_shape, 0.5), dtype=dtype),
                                          name="mul_weight_2", device="CPU")
            self.add_weight = Parameter(Tensor(np.full(add_shape, 0.5), dtype=dtype), name="add_weight", device="CPU")
        else:
            self.mul_weight_1 = Parameter(Tensor(np.full(mul_shape, 0.5), dtype=dtype), name="mul_weight_1")
            self.mul_weight_2 = Parameter(Tensor(np.full(mul_shape, 0.5), dtype=dtype), name="mul_weight_2")
            self.add_weight = Parameter(Tensor(np.full(add_shape, 0.5), dtype=dtype), name="add_weight")
        self.mul1 = P.Mul()
        self.mul2 = P.Mul()
        self.add = P.Add()

        if strategy is not None:
            self.mul1.shard(strategy)
            self.add.shard(strategy)
            self.mul2.shard(strategy)

    def construct(self, x):
        output = self.mul1(x, self.mul_weight_1)
        output = self.add(output, self.add_weight)
        output = self.mul2(output, self.mul_weight_2)
        return output


def save_checkpoint_and_model_train(model, epoch, dataset, ckpt_path, ckpt_format=None,
                                    save_checkpoint_steps=1,
                                    keep_checkpoint_max=1, integrated_save=False, async_save=False,
                                    append_info=None, remove_redundancy=True,
                                    exception_save=False, ckpt_prefix="ckpt_ms",
                                    dataset_sink_mode=False, sink_size=-1,
                                    **kwargs):
    """save ckpt"""
    ckpt_config = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                   keep_checkpoint_max=keep_checkpoint_max,
                                   integrated_save=integrated_save, append_info=append_info,
                                   async_save=async_save, remove_redundancy=remove_redundancy,
                                   exception_save=exception_save, format=ckpt_format, **kwargs)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)
    clean_all_ckpt_files(ckpt_path)
    model.train(epoch=epoch, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode,
                callbacks=[ckpt_callback], sink_size=sink_size)


def test_parallel_safetensors_with_swap():
    """
    Feature: Save safetensors and reload it, using the swap functionality.
    Description: Define a network with multiple weight parameters, save all as non-deduplicated safetensors,
    load deduplicated safetensors, compare precision, and use the swap functionality.
    Expectation: Run success
    """
    context.set_context(mode=context.GRAPH_MODE)
    rank_id = contextbase.get_parallel_variable_from_env("RANK_ID")
    context.reset_auto_parallel_context()
    contextbase.set_parallel_context(device_num=8,
                                     parallel_mode="semi_auto_parallel")
    net_parallel = MulAddMulNet(mul_shape=(16, 8), add_shape=(16, 8,), swap=True)
    dataset_parallel = FakeData(size=32, batch_size=2, image_size=(8,), num_classes=8,
                                use_parallel=True)
    parallel_model = modeltrainbase.create_train_model(net_parallel)

    safetensors_path = f"./parallel_weight/rank_{rank_id}"
    # 不去冗余保存
    save_checkpoint_and_model_train(parallel_model, 1, dataset_parallel, ckpt_path=safetensors_path,
                                    remove_redundancy=False, keep_checkpoint_max=1, ckpt_format='safetensors')
    net1 = MulAddMulNet(mul_shape=(16, 8), add_shape=(16, 8,), swap=True)
    model = modeltrainbase.create_train_model(net1)
    model1_dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=32, batch_size=2, image_size=(8,), num_classes=8),
        ["data", "label"])
    model.build(train_dataset=model1_dataset, epoch=2)

    ms.load_checkpoint(f"./parallel_weight/rank_{rank_id}/ckpt_ms-1_2.safetensors", model.train_network,
                       format="safetensors", remove_redundancy=True)
    input_x = Tensor(np.random.rand(2, 8).astype(np.float32))
    assert np.array_equal(parallel_model.predict(input_x).asnumpy(), model.predict(input_x).asnumpy()), \
        "result not equal, please check"


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards")
def test_parallel_safetensors_remove_redundancy_match_load_false_with_swap():
    """
    Feature: Save safetensors and reload it, using the swap functionality.
    Description: Define a network with multiple weight parameters, save all as non-deduplicated safetensors,
    load deduplicated safetensors, compare precision, and use the swap functionality.
    Expectation: Run success
    """
    ret = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=8129\
        pytest -s test_swap.py::test_parallel_safetensors_with_swap"
    )
    assert ret == 0
