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
"""Test custom communication group creation and reuse in distributed training.

Validates that custom communication groups are correctly created and reused by the
parallel framework, and that standalone and parallel training produce consistent results.
"""
import re
import os
import numpy as np
import subprocess
import copy
import mindspore as ms
from mindspore import ops
from mindspore import nn, Parameter, Tensor
from mindspore.communication.management import init
from mindspore.communication.management import create_group
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from tests.st.auto_parallel.model_parallel import FakeData
from tests.st.auto_parallel.utils.utils import _count_unequal_element
from tests.st.auto_parallel.utils.modeltrain_base import modeltrainbase


def setup_function():
    """Initialize the test environment with context and communication management.
    
    This setup function is automatically called before running the tests in this module.
    It sets MindSpore to GRAPH_MODE (0), enables graph saving for debugging, and 
    initializes the distributed communication framework.
    """
    ms.set_context(mode=0)
    ms.set_context(save_graphs=True, save_graphs_path="./parallel_group_reuse/ir")
    init()


class CompareBase():
    """Base class for comparing numerical arrays and checkpoint files.
    
    This class provides utility methods for comparing:
    - NumPy arrays with specified tolerance levels
    - Checkpoint files by comparing model outputs after loading checkpoints
    - Checkpoint dictionaries by comparing network outputs after parameter loading
    
    Used extensively in testing to validate that different training configurations
    (standalone vs parallel) produce numerically consistent results.
    """
    def compare_nparray(self, data_expected, data_me, rtol, atol, equal_nan=True):
        """Compare two NumPy arrays with relative and absolute tolerance.
        
        Args:
            data_expected: Expected NumPy array.
            data_me: Actual NumPy array to compare.
            rtol: Relative tolerance threshold.
            atol: Absolute tolerance threshold.
            equal_nan: Whether to consider NaN values as equal. Default is True.
            
        Raises:
            AssertionError: If arrays are not close within the specified tolerance,
                          or if array shapes don't match.
        """
        if np.any(np.isnan(data_expected)):
            assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
        elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
            _count_unequal_element(data_expected, data_me, rtol, atol)
        else:
            assert np.array(data_expected).shape == np.array(data_me).shape

    def compare_checkpoint(self, expected_net, ckpt_expected, ckpt_actual, inputdata, rtol, atol):
        """Compare two checkpoint files by comparing network outputs after loading.
        
        This method loads checkpoint files into networks and compares their outputs
        on the same input data, verifying that the checkpoints produce equivalent results.
        
        Args:
            expected_net: Reference neural network model.
            ckpt_expected: Path to the expected checkpoint file.
            ckpt_actual: Path to the actual checkpoint file to verify.
            inputdata: Input tensor for network inference.
            rtol: Relative tolerance threshold for output comparison.
            atol: Absolute tolerance threshold for output comparison.
        """
        data_expected = load_checkpoint(ckpt_expected)
        data_actual = load_checkpoint(ckpt_actual)
        actual_net = copy.deepcopy(expected_net)

        load_param_into_net(expected_net, data_expected)
        expected_out = expected_net(inputdata)

        load_param_into_net(actual_net, data_actual)
        actual_out = actual_net(inputdata)

        self.compare_nparray(expected_out.asnumpy(), actual_out.asnumpy(), rtol, atol)

    def compare_checkpoint_dict(self, expected_net, ckpt_expected, ckpt_actual, *inputs,
                                rtol=0.001, atol=0.001):
        """Compare two checkpoint dictionaries by comparing network outputs.
        
        This method loads checkpoint dictionaries into networks and compares their outputs
        on the provided input data, supporting multiple input tensors.
        
        Args:
            expected_net: Reference neural network model.
            ckpt_expected: Expected checkpoint dictionary.
            ckpt_actual: Actual checkpoint dictionary to verify.
            *inputs: Variable number of input tensors for network inference.
            rtol: Relative tolerance threshold for output comparison. Default is 0.001.
            atol: Absolute tolerance threshold for output comparison. Default is 0.001.
        """
        actual_net = copy.deepcopy(expected_net)
        load_param_into_net(expected_net, ckpt_expected)
        expected_out = expected_net(*inputs)
        load_param_into_net(actual_net, ckpt_actual)
        actual_out = actual_net(*inputs)
        self.compare_nparray(expected_out.asnumpy(), actual_out.asnumpy(), rtol, atol)




class Net1(nn.Cell):
    """A simple neural network combining addition and matrix multiplication operations.
    
    This network performs element-wise addition with a learnable bias weight,
    followed by matrix multiplication with another learnable weight. Both operations
    support sharding strategies for distributed training and are used to test
    communication group reuse in parallel scenarios.
    
    Attributes:
        add: Element-wise addition operation.
        matmul: Matrix multiplication operation.
        matmul_weight: Learnable weight parameter for matrix multiplication.
        add_weight: Learnable weight parameter for addition.
    """
    def __init__(self, weight_shape, in_strategy=None):
        """Initialize the Net1 network.
        
        Args:
            weight_shape: Shape tuple for both weight parameters.
            in_strategy: Optional sharding strategy for both add and matmul operations.
                        Default is None.
        """
        super().__init__()
        self.add = ops.Add()
        self.matmul = ops.MatMul()
        self.matmul_weight = Parameter(Tensor(np.ones(weight_shape).astype(np.float32)), name="matmul_weight")
        self.add_weight = Parameter(Tensor(np.ones(weight_shape).astype(np.float32)), name="add_weight")
        self.add.shard(in_strategy=in_strategy)
        self.matmul.shard(in_strategy=in_strategy)

    def construct(self, input1, label):
        """Forward pass applying addition and matrix multiplication.
        
        Args:
            input: Input1 tensor.
            label: Label tensor (unused, passed for compatibility).
            
        Returns:
            Output tensor after applying addition and matrix multiplication sequentially.
        """
        x = self.add(input1, self.add_weight)
        x = self.matmul(x, self.matmul_weight)
        return x


def check_ir(rank_id, rank_list, group_name, ir_path="./parallel_group_reuse/ir", exist_flag=True):
    """Verify that a communication group name exists (or doesn't exist) in IR graphs.
    
    This function checks the generated IR (intermediate representation) files for a specific
    rank to verify whether a communication group with a given name is present or absent.
    This is used to validate that custom communication groups are correctly created and
    used during parallel training.
    
    Args:
        rank_id: Current rank ID in the distributed environment.
        rank_list: List of rank IDs that should have the IR file.
        group_name: Name of the communication group to search for in IR files.
        ir_path: Path to the directory containing IR files. Default is "ir".
        exist_flag: If True, asserts that the group exists; if False, asserts it doesn't exist.
                   Default is True.
                   
    Raises:
        AssertionError: If the group existence doesn't match the expect_flag.
    """
    if rank_id in rank_list:
        current_directory = os.getcwd()
        ir_path = os.path.join(current_directory, f"{ir_path}", f"rank_{rank_id}")
        result = subprocess.run(
            ['grep', '-nr', 'group: "', f'{ir_path}'],
            capture_output=True,
            text=True,
            check=False)
        # 输出结果
        obj = result.stdout
        result = re.search(group_name, obj)
        if exist_flag:
            assert result is not None
        else:
            assert result is None

def save_graphs_func(save_graphs_flag=0, save_graphs_path="./parallel_group_reuse/"):
    """Configure environment variables for saving computation graphs during compilation.
    
    This utility function sets environment variables to control whether MindSpore should
    save intermediate representation (IR) graphs and where they should be saved.
    
    Args:
        save_graphs_flag: Integer flag for graph saving mode. Default is 0 (disabled).
        save_graphs_path: Directory path where graph files will be saved. Default is current directory.
    """
    os.environ['MS_DEV_SAVE_GRAPHS'] = str(save_graphs_flag)
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = save_graphs_path

def test_parallel_creat_group_reuse_001():
    """Test custom communication group creation and reuse in distributed training.
    
    This comprehensive test validates that:
    1. Custom communication groups can be created for specific rank subsets
    2. These custom groups are correctly reused by the parallel framework instead of
       creating new default groups
    3. Standalone and parallel training produce numerically consistent results
    4. The IR graphs correctly reference the custom communication groups
    
    The test performs the following phases:
    Phase 1: Create custom communication groups for rank pairs (0-1) and (0-2)
    Phase 2: Train a standalone network without parallelization
    Phase 3: Train the same network with parallel configuration and sharding strategy
    Phase 4: Verify checkpoint equivalence by comparing network outputs
    Phase 5: Verify that custom groups appear in the IR graphs
    
    Test setup:
    - 4 devices total (assuming at least 4 ranks in test environment)
    - Custom group 1: ranks 0 and 1
    - Custom group 2: ranks 0 and 2
    - Network: Net1 with 8x8 weight matrices
    - Sharding strategy: (2,2) for both add and matmul operations
    """
    # Initialize distributed communication
    D.init()
    rank_id = get_rank()
    # Enable IR graph saving for verification
    save_graphs_func(save_graphs_flag=1, save_graphs_path="./parallel_group_reuse/ir")

    # Define custom communication groups
    rank_ids0 = [0, 1]
    rank_ids1 = [0, 2]
    group_name0 = "customed groups 0-1"
    group_name1 = "customed groups 0-2"

    # Create custom groups for specific rank pairs
    if rank_id in rank_ids0:
        create_group(group_name0, rank_ids0)
    if rank_id in rank_ids1:
        create_group(group_name1, rank_ids1)

    # Initialize after group creation
    init()

    # ===== Phase 2: Standalone training (no parallelization) =====
    standalone_net = Net1(weight_shape=(8, 8))
    standalone_dataset = FakeData(size=16, batch_size=8, image_size=(8,), num_classes=8)
    standalone_model = modeltrainbase.create_train_model(standalone_net, loss=None)
    standalone_ckpt = modeltrainbase.load_newest_ckpt_from_model_train(
        standalone_model, epoch=2, dataset=standalone_dataset, dataset_sink_mode=False,
        ckpt_path="./parallel_group_reuse/rank_{}_ckpt".format(rank_id),
        ckpt_prefix="ckpt_standalone", load_format="name")

    # ===== Phase 3: Parallel training with sharding strategy =====
    in_strategy = ((2, 2), (2, 2))
    net = Net1(weight_shape=(8, 8), in_strategy=in_strategy)
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_dataset = FakeData(size=16, batch_size=1, image_size=(8,), use_parallel=True, num_classes=8)
    parallel_model = modeltrainbase.create_train_model(parallel_net, loss=None)
    parallel_ckpt = modeltrainbase.load_newest_ckpt_from_model_train(
        parallel_model, epoch=2, dataset=parallel_dataset, dataset_sink_mode=False,
        ckpt_path="./parallel_group_reuse/rank_{}_ckpt".format(rank_id),
        ckpt_prefix="ckpt_parallel", load_format="name")

    # ===== Phase 4: Verify checkpoint equivalence =====
    net_cmp = Net1(weight_shape=(8, 8))
    input_x = Tensor(np.random.randn(8, 8).astype(np.float32))
    input_y = Tensor(np.random.randn(1, 1).astype(np.float32))

    # Compare outputs from standalone and parallel checkpoints
    comparebase = CompareBase()
    comparebase.compare_checkpoint_dict(net_cmp, standalone_ckpt, parallel_ckpt, input_x, input_y)

    # ===== Phase 5: Verify custom groups in IR graphs =====
    check_ir(rank_id, [0, 1], group_name0)
    check_ir(rank_id, [0, 2], group_name1)
