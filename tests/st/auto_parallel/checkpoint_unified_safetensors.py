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
"""Test unified safetensors checkpoint saving with pipeline parallelism.

This module tests the end-to-end workflow of:
1. Pipeline parallelism across 8 devices with 2 pipeline stages
2. Network creation with shared parameters (MatMulStageNet)
3. Checkpoint saving in safetensors format with parallel optimizer
4. Multi-rank checkpoint synchronization
5. Unified checkpoint consolidation from all ranks

The module validates that:
- Shared parameters are correctly managed across pipeline stages
- Safetensors format provides efficient checkpoint storage
- Multi-rank checkpoints are properly synchronized before unification
- Unified checkpoints can be loaded and used for inference

Key components:
- MatMulNet: Simple matrix multiplication operation cell
- MatMulStageNet: Multi-stage network with shared parameters for pipeline parallelism
- Utility functions for checkpoint synchronization and cleanup
- Test function for validating the complete workflow
"""
import os
import time
import numpy as np
import mindspore as ms
from mindspore import context, lazy_inline, Parameter, Tensor, unified_safetensors
from mindspore import log as logger
from mindspore.nn import Cell
from mindspore.communication.management import init
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
import mindspore.ops.operations as P
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from tests.st.auto_parallel.model_parallel import FakeData
from tests.st.auto_parallel.utils.modeltrain_base import modeltrainbase


def setup_function():
    """Initialize the test environment with context and communication management.
    
    This setup function is automatically called before running the tests in this module.
    It sets MindSpore to GRAPH_MODE (0), enables graph saving for debugging, and 
    initializes the distributed communication framework.
    """
    ms.set_context(mode=0)
    ms.set_context(save_graphs=True, save_graphs_path="./test_checkpoint_unified/ir")
    init()


class MatMulNet(Cell):
    """A simple neural network cell that performs matrix multiplication operations.
    
    This class encapsulates a matrix multiplication operation that can be sharded
    across multiple devices according to a specified strategy. It is used as a
    building block for more complex network architectures in distributed training scenarios.
    """

    def __init__(self, strategy=None):
        """Initialize the MatMulNet cell.
        
        Args:
            strategy: Optional sharding strategy for the matrix multiplication operation.
                     If provided, the strategy is applied to the matmul1 operation.
                     Default is None, meaning no sharding is applied.
        """
        super().__init__()
        self.matmul1 = P.MatMul()
        if strategy is not None:
            self.matmul1.shard(strategy)

    def construct(self, inputs, label):
        """Forward pass of the MatMulNet cell.
        
        Args:
            inputs: Input tensor for the matrix multiplication.
            label: Label or weight tensor for the matrix multiplication.
            
        Returns:
            The result of matrix multiplication between inputs and label.
        """
        x = self.matmul1(inputs, label)
        return x


class MatMulStageNet(Cell):
    """A multi-stage neural network with shared parameters for pipeline parallelism.
    
    This network consists of two MatMulNet layers that share a common weight parameter.
    It is designed to be used with pipeline parallelism where different stages can be
    assigned to different devices, and checkpoints can be saved in safetensors format.
    
    Attributes:
        matmul_weight: A shared parameter tensor used by both layers.
        layer1: First MatMulNet layer for initial processing.
        layer2: Second MatMulNet layer for further processing.
    """

    @lazy_inline
    def __init__(self, matmul_weight, strategy=None):
        """Initialize the MatMulStageNet with shared parameters.
        
        Args:
            matmul_weight: A tensor that serves as the shared weight parameter for both layers.
            strategy: Optional sharding strategy to apply to both MatMulNet layers.
                     Default is None.
        """
        super().__init__()
        self.matmul_weight = Parameter(matmul_weight, name="weight2")
        self.layer1 = MatMulNet(strategy)
        self.layer2 = MatMulNet(strategy)

    def construct(self, inputs, label):
        """Forward pass through both layers using shared parameters.
        
        Args:
            inputs: Input tensor for the first layer.
            label: Label tensor for the first layer (unused, passed for compatibility).
            
        Returns:
            Output tensor after passing through both MatMulNet layers sequentially.
        """
        x = self.layer1(inputs, self.matmul_weight)
        x = self.layer2(x, self.matmul_weight)
        return x


def check_checkpoint_file_by_rank(rank_id, global_rank, dst_path):
    """Synchronize checkpoint file writes across distributed ranks.
    
    This function ensures that all distributed ranks have completed writing their
    checkpoint files before proceeding. It uses a file-based synchronization mechanism
    where each rank waits for checkpoint files from all other ranks.
    
    Args:
        rank_id: The current rank's ID in the distributed training setup.
        global_rank: The total number of ranks in the distributed environment.
        dst_path: The destination path where checkpoint files are being saved.
        
    Returns:
        None
    """
    file_path = f"./{dst_path}/rank_{rank_id}"
    start = time.time()
    while not os.path.exists(file_path):
        time.sleep(1)
        end = time.time()
        if end > start + 10:
            break
    with open(f"{file_path}/change_end{rank_id}.txt", "w", encoding='utf-8') as f:
        f.write("change end")
    for i in range(global_rank):
        while not os.path.exists(f"./{dst_path}/rank_{i}/change_end{i}.txt"):
            time.sleep(1)
            end = time.time()
            if end > start + 10:
                break


def clean_all_ckpt_files(folder_path):
    """Remove all checkpoint and metadata files from a specified directory.
    
    This utility function cleans up checkpoint files (*.ckpt) and metadata files (*.meta)
    from the specified folder. It is useful for removing old checkpoints before training
    to ensure a clean state. Any FileNotFoundError exceptions are logged as warnings
    and do not interrupt the cleanup process.
    
    Args:
        folder_path: The directory path containing checkpoint files to be cleaned.
        
    Returns:
        None
    """
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ckpt file error.".format(e))


def save_checkpoint_and_model_train(model, epoch, dataset, ckpt_path, format_=None,
                                    save_checkpoint_steps=1,
                                    keep_checkpoint_max=1, integrated_save=False, async_save=False,
                                    append_info=None,
                                    exception_save=False, ckpt_prefix="ckpt_ms",
                                    dataset_sink_mode=False, sink_size=-1, remove_redundancy=False,
                                    **kwargs):
    """Train a model and save checkpoints in the specified format.
    
    This function combines model training with checkpoint saving functionality. It creates
    a checkpoint callback with the specified configuration, cleans up old checkpoint files,
    and then trains the model while periodically saving checkpoints according to the
    configured strategy.
    
    Args:
        model: The MindSpore model to train.
        epoch: Number of epochs to train the model.
        dataset: The training dataset to use for training.
        ckpt_path: The directory path where checkpoints will be saved.
        format: Checkpoint format to use. Default is None (uses default format).
                Can be 'safetensors', 'ckpt', etc.
        save_checkpoint_steps: Number of training steps between checkpoint saves. Default is 1.
        keep_checkpoint_max: Maximum number of checkpoint files to keep. Default is 1.
        integrated_save: Whether to use integrated save for distributed training. Default is False.
        async_save: Whether to save checkpoints asynchronously. Default is False.
        append_info: Additional information to append to checkpoint files. Default is None.
        exception_save: Whether to save checkpoint on exception. Default is False.
        ckpt_prefix: Prefix for checkpoint file names. Default is "ckpt_ms".
        dataset_sink_mode: Whether to use dataset sink mode. Default is False.
        sink_size: Number of steps to sink. Default is -1 (all steps).
        remove_redundancy: Whether to remove redundant checkpoint data. Default is False.
        **kwargs: Additional keyword arguments to pass to CheckpointConfig.
        
    Returns:
        None
    """
    ckpt_config = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                   keep_checkpoint_max=keep_checkpoint_max,
                                   integrated_save=integrated_save, append_info=append_info,
                                   async_save=async_save,
                                   exception_save=exception_save, format=format_, remove_redundancy=remove_redundancy,
                                   **kwargs)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)

    clean_all_ckpt_files(ckpt_path)

    model.train(epoch=epoch, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode,
                callbacks=[ckpt_callback], sink_size=sink_size)


def test_unified_safetensors_pp_shared_param():
    """Test unified safetensors checkpoint saving with pipeline parallelism and shared parameters.
    
    This test validates the end-to-end workflow of:
    1. Initializing distributed communication across 8 devices with 2 pipeline stages
    2. Creating a network with shared parameters (MatMulStageNet) 
    3. Assigning layers to different pipeline stages (stage 0 and stage 1)
    4. Training the network with pipeline parallelism
    5. Saving checkpoints in safetensors format with parallel optimizer enabled
    6. Synchronizing checkpoint files across all ranks
    7. Unifying and consolidating checkpoints from all ranks into a single checkpoint
    
    The test ensures that:
    - Shared parameters are correctly saved and managed across pipeline stages
    - Checkpoints can be saved in safetensors format (more efficient than pickle-based ckpt)
    - Multi-rank checkpoint files are properly synchronized before unification
    - The unified checkpoint can be loaded and used for inference/fine-tuning
    
    Raises:
        Various exceptions if distributed training setup, checkpoint saving, or unification fails.
    """
    # Initialize distributed communication
    D.init()
    rank_id = get_rank()
    # Configure automatic parallel context for distributed training
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, pipeline_stages=2,
                                      parallel_mode="semi_auto_parallel",
                                      strategy_ckpt_config={
                                          "save_file": f"./test_checkpoint_unified/strategy_{rank_id}.ckpt"},
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={'parallel_optimizer_threshold': 0})

    # Create network with shared weight parameter
    matmul_weight = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    net_parallel = MatMulStageNet(matmul_weight, strategy=((2, 1), (1, 2)))
    # Assign layers to pipeline stages
    net_parallel.layer1.pipeline_stage = 0
    net_parallel.layer2.pipeline_stage = 1

    # Create dataset for distributed training
    dataset_parallel = FakeData(size=16, batch_size=2, image_size=(16,), num_classes=8,
                                use_parallel=True)

    # Wrap network with pipeline cell for pipeline parallelism execution
    net = PipelineCell(net_parallel, 2)
    parallel_model = modeltrainbase.create_train_model(net, loss=None)

    # Train and save checkpoints in safetensors format
    safetensors_path = f"./test_checkpoint_unified/parallel_weight/rank_{rank_id}"
    save_checkpoint_and_model_train(parallel_model, 1, dataset_parallel, ckpt_path=safetensors_path,
                                    keep_checkpoint_max=1, format_='safetensors')

    # Synchronize all ranks before unification
    check_checkpoint_file_by_rank(rank_id, 8, "./test_checkpoint_unified/parallel_weight")

    # Rank 0 unifies checkpoints from all ranks into a single unified checkpoint
    if rank_id == 0:
        unified_safetensors("./test_checkpoint_unified/parallel_weight",
                            f"./test_checkpoint_unified/strategy_{rank_id}.ckpt",
                            "./test_checkpoint_unified/unified_safetensors")
