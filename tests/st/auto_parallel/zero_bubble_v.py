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
"""Test zero-bubble pipeline parallelism strategy with multi-stage pipeline execution.

Tests 4-stage pipeline parallelism with zero-bubble-v scheduler, checkpoint transformation,
and distributed inference validation across multiple ranks.
"""
import os
import time
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
from mindspore import nn
from mindspore import lazy_inline
from mindspore import ParameterTuple
from mindspore import Tensor
from mindspore import load_param_into_net
from mindspore.parallel import merge_pipeline_strategys
from mindspore.communication import init
from mindspore.nn import Momentum
from mindspore.train import Model
from mindspore.common.parameter import Parameter
from mindspore.parallel.nn import Pipeline
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.train.serialization import load_checkpoint
from mindspore.parallel.checkpoint_transform import transform_checkpoints
import mindspore.communication.management as D
from .utils.modeltrain_base import modeltrainbase
from .model_parallel import FakeData


class WithLossCell(nn.Cell):
    """A wrapper cell combining a backbone network with a loss function.
    
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


class MatMulNet(nn.Cell):
    """A neural network cell with two sequential matrix multiplication operations.
    
    This network applies two matrix multiplications with learnable weight parameters.
    Both operations support optional sharding strategies for distributed training.
    It is used as a building block in pipeline stages for testing zero-bubble pipeline strategies.
    
    Attributes:
        matmul1: First matrix multiplication operation.
        matmul2: Second matrix multiplication operation.
        matmul1_weight: Learnable weight for the first matmul operation.
        matmul2_weight: Learnable weight for the second matmul operation.
    """
    def __init__(self, matmul_weight, **kwargs):
        """Initialize the MatMulNet with weight tensors and optional sharding strategies.
        
        Args:
            matmul_weight: List of two weight tensors for matmul1 and matmul2.
            **kwargs: Optional keyword arguments:
                - strategy1: Sharding strategy for matmul1 operation. Default is None.
                - strategy2: Sharding strategy for matmul2 operation. Default is None.
        """
        super().__init__()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()
        self.matmul1_weight = Parameter(matmul_weight[0], name="weight1")
        self.matmul2_weight = Parameter(matmul_weight[1], name="weight2")
        if "strategy1" in kwargs and kwargs["strategy1"] is not None:
            self.matmul1.shard(kwargs["strategy1"])
        if "strategy2" in kwargs and kwargs["strategy2"] is not None:
            self.matmul2.shard(kwargs["strategy2"])

    def construct(self, inputs):
        """Forward pass applying two sequential matrix multiplications.
        
        Args:
            inputs: Input tensor.
            
        Returns:
            Output tensor after two matrix multiplications.
        """
        x = self.matmul1(inputs, self.matmul1_weight)
        x = self.matmul2(x, self.matmul2_weight)
        return x



class StageNet(nn.Cell):
    """A multi-stage neural network with multiple MatMulNet blocks for pipeline testing.
    
    This network consists of multiple MatMulNet blocks, each followed by ReLU activation
    and bias addition. It is designed to test zero-bubble pipeline strategies with
    multiple pipeline stages and segment configurations.
    
    Attributes:
        micro_size: Number of micro-batches or stages in the network.
        block: CellList containing MatMulNet blocks.
        relu_block: CellList containing ReLU activation layers.
        add_list: List of bias parameters for addition operations.
        add_tuple: ParameterTuple of all bias parameters.
    """
    def __init__(self, weight_list, micro_size, **kwargs):
        """Initialize the StageNet with multiple blocks.
        
        Args:
            weight_list: List of weight tensors for each MatMulNet block.
            micro_size: Number of blocks/stages in the network.
            **kwargs: Optional keyword arguments passed to MatMulNet:
                - strategy1: Sharding strategy for matmul1. Default is None.
                - strategy2: Sharding strategy for matmul2. Default is None.
        """
        super().__init__()
        self.micro_size = micro_size
        self.block = nn.CellList()
        self.add = P.TensorAdd()
        self.weight_list = weight_list
        self.add_list = []
        self.relu_block = nn.CellList()
        for i in range(self.micro_size):
            cell = MatMulNet(weight_list[i], **kwargs)
            relu = nn.ReLU()
            self.relu_block.append(relu)
            self.block.append(cell)
            self.add_list.append(
                Parameter(Tensor(np.full((1, 16), 0.1, dtype=np.float32)), name=f"weight{i}"))
        self.add_tuple = ParameterTuple(self.add_list)

    def construct(self, x):
        """Forward pass through all stages with MatMul, ReLU, and bias addition.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after processing through all stages sequentially.
        """
        for i in range(self.micro_size):
            x = self.block[i](x)
            x = self.relu_block[i](x)
            x = self.add(x, self.add_tuple[i])
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


def predict_each_rank(net, ckpt_file, predict_data):
    """Execute model prediction after loading checkpoint for each rank.
    
    This function loads a checkpoint into a network model and performs inference
    on the provided data. It is used to validate that different network configurations
    produce consistent results after strategy transformation.
    
    Args:
        net: The neural network model for inference.
        ckpt_file: Path to the checkpoint file to load.
        predict_data: Input tensor for prediction.
        
    Returns:
        Inference result tensor from the model.
    """
    model = Model(net)
    model.infer_predict_layout(predict_data)
    predict_ckpt = load_checkpoint(ckpt_file)
    load_param_into_net(net, predict_ckpt, strict_load=True)
    infer = model.predict(predict_data)
    return infer


def setup_function():
    """Initialize the test environment with context and communication management.
    
    This setup function is automatically called before running the tests in this module.
    It sets MindSpore to GRAPH_MODE (0), enables graph saving for debugging, and 
    initializes the distributed communication framework.
    """
    ms.set_context(mode=0)
    ms.set_context(save_graphs=True, save_graphs_path="./zero_bubble/ir")
    init()


def _count_unequal_element(data_expected, data_me, rtol, atol):
    """Count and analyze unequal elements between two arrays within tolerance.
    
    This utility function compares two arrays element-wise and counts how many elements
    exceed the specified relative and absolute tolerance thresholds. It also handles
    special cases like NaN and infinity values. Used for detailed error analysis when
    numerical comparisons fail.
    
    Args:
        data_expected: Expected NumPy array.
        data_me: Actual NumPy array to compare.
        rtol: Relative tolerance threshold.
        atol: Absolute tolerance threshold.
        
    Raises:
        AssertionError: If the loss count exceeds the relative tolerance ratio.
    """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    # ICKTGQ
    if data_expected.dtype in ('complex64', 'complex128'):
        greater = greater + nan_diff + inf_diff
    else:
        neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
        greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """Compare two arrays for approximate equality within specified tolerances.
    
    This function checks if two NumPy arrays are approximately equal within the specified
    relative and absolute tolerance thresholds. If comparison fails, it provides detailed
    error analysis of which elements differ and by how much.
    
    Args:
        data_expected: Expected NumPy array.
        data_me: Actual NumPy array to compare.
        rtol: Relative tolerance threshold.
        atol: Absolute tolerance threshold.
        equal_nan: Whether to consider NaN values as equal. Default is True.
    """
    if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def test_pipeline_zero_bubble_v_001():
    '''
    Feature: Pipeline zero bubble v.
    Description: Test zero bubble v.
    Expectation: Run success.
    '''
    # Initialize distributed communication
    D.init()
    rank_id = D.get_rank()
    np.random.seed(100)

    # Create weight tensors for multiple stages
    weight1 = Tensor(0.1 * np.random.randn(96, 16).astype(np.float32))
    weight2 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight3 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight4 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight5 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight6 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight7 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight8 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight_list = [[weight1, weight2], [weight3, weight4], [weight5, weight6], [weight7, weight8]]
    predict_data = Tensor(np.random.randn(16, 96).astype(np.float32))

    # ===== Phase 1: Pipeline training with segment and zero-bubble-v scheduler =====
    net = StageNet(weight_list=weight_list, micro_size=4, strategy1=((2, 1), (1, 1)), strategy2=((1, 1), (1, 2)))
    loss_fn = nn.L1Loss()
    net_with_loss = Pipeline(WithLossCell(net, loss_fn), 4,
                             stage_config={'_backbone.block.0': 0, '_backbone.relu_block.0': 0,
                                           '_backbone.block.1': 1, '_backbone.relu_block.1': 1,
                                           '_backbone.block.2': 1, '_backbone.relu_block.2': 1,
                                           '_backbone.block.3': 0, '_backbone.relu_block.3': 0},
                             segment_config={'_backbone.block.0': 0, '_backbone.relu_block.0': 0,
                                             '_backbone.block.1': 0, '_backbone.relu_block.1': 0,
                                             '_backbone.block.2': 1, '_backbone.relu_block.2': 1,
                                             '_backbone.block.3': 1, '_backbone.relu_block.3': 1})
    parallel_net = AutoParallel(net_with_loss, parallel_mode="semi_auto")
    parallel_net.pipeline(stages=2, interleave=True, scheduler='zero_bubble_v')
    parallel_net.save_param_strategy_file(f"./zero_bubble/pipe_strategy/pipeline_segment_{rank_id}.ckpt")
    pipeline_dataset = FakeData(size=64, batch_size=16, image_size=(96,), num_classes=16)
    opt = Momentum(learning_rate=0.00001, momentum=0.09, params=parallel_net.trainable_params())
    pipeline_model = modeltrainbase.create_train_model(parallel_net, loss=None, opt=opt)
    modeltrainbase.load_newest_ckpt_from_model_train(pipeline_model, epoch=1,
                                                     dataset=pipeline_dataset,
                                                     dataset_sink_mode=False,
                                                     integrated_save=False,
                                                     format_="safetensors",
                                                     ckpt_path=f"./zero_bubble/pipe/rank_{rank_id}",
                                                     ckpt_prefix="pipeline")

    # ===== Phase 2: Non-pipeline training with different sharding strategy =====
    net = StageNet(weight_list=weight_list, micro_size=4, strategy1=((2, 1), (1, 2)), strategy2=((2, 1), (1, 2)))
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_net.save_param_strategy_file("./zero_bubble/parallel.ckpt")
    parallel_dataset = FakeData(size=64, batch_size=16, image_size=(96,), num_classes=16)
    opt = Momentum(learning_rate=0.00001, momentum=0.09, params=parallel_net.trainable_params())
    _ = modeltrainbase.create_model_and_train(parallel_net, dataset=parallel_dataset, loss=loss_fn,
                                                           opt=opt, dataset_sink_mode=False)

    # ===== Phase 3: Pipeline training without segment configuration =====
    net1 = StageNet(weight_list=weight_list, micro_size=4, strategy1=((2, 1), (1, 2)), strategy2=((2, 1), (1, 2)))
    net_with_loss = Pipeline(WithLossCell(net1, loss_fn), 4,
                             stage_config={'_backbone.block.0': 0, '_backbone.relu_block.0': 0,
                                           '_backbone.block.1': 0, '_backbone.relu_block.1': 0,
                                           '_backbone.block.2': 1, '_backbone.relu_block.2': 1,
                                           '_backbone.block.3': 1, '_backbone.relu_block.3': 1})
    pipeline_net = AutoParallel(net_with_loss, parallel_mode="semi_auto")
    pipeline_net.pipeline(stages=2)
    pipeline_net.save_param_strategy_file(f"./zero_bubble/pipeline/pipeline{rank_id}.ckpt")
    pipeline_dataset = FakeData(size=64, batch_size=16, image_size=(96,), num_classes=16)
    opt = Momentum(learning_rate=0.00001, momentum=0.09, params=pipeline_net.trainable_params())
    pipeline_model = modeltrainbase.create_model_and_train(pipeline_net, dataset=pipeline_dataset, loss=None,
                                                           opt=opt, dataset_sink_mode=False)

    # ===== Phase 4: Transform checkpoints (only rank 0 does the transformation) =====
    if rank_id == 0:
        # Merge strategy files
        merge_pipeline_strategys("./zero_bubble/pipe_strategy", "./zero_bubble/pipe_strategy/pipe_strategy.ckpt")
        merge_pipeline_strategys("./zero_bubble/pipeline", "./zero_bubble/pipeline/pipeline.ckpt")

        # Transform pipeline checkpoints to parallel format
        transform_checkpoints(src_checkpoints_dir="./zero_bubble/pipe",
                              dst_checkpoints_dir="./zero_bubble/change",
                              ckpt_prefix="parallel_changed",
                              src_strategy_file="./zero_bubble/pipe_strategy/pipe_strategy.ckpt",
                              dst_strategy_file="./zero_bubble/parallel.ckpt")

        # Transform pipeline checkpoints to pipeline format without segment
        transform_checkpoints(src_checkpoints_dir="./zero_bubble/pipe",
                              dst_checkpoints_dir="./zero_bubble/change_pipe",
                              ckpt_prefix="pipeline_changed",
                              src_strategy_file="./zero_bubble/pipe_strategy/pipe_strategy.ckpt",
                              dst_strategy_file="./zero_bubble/pipeline/pipeline.ckpt")

    # ===== Phase 5: Synchronize checkpoint files across all ranks =====
    check_checkpoint_file_by_rank(rank_id, 8, "./zero_bubble/change")
    check_checkpoint_file_by_rank(rank_id, 8, "./zero_bubble/change_pipe")
    init()

    # ===== Phase 6: Inference with transformed pipeline checkpoints =====
    net_pipe = StageNet(weight_list=weight_list, micro_size=4, strategy1=((2, 1), (1, 2)), strategy2=((2, 1), (1, 2)))
    net_with_loss = Pipeline(net_pipe, 1,
                             stage_config={'block.0': 0, 'relu_block.0': 0, 'block.1': 0, 'relu_block.1': 0,
                                           'block.2': 1, 'relu_block.2': 1, 'block.3': 1, 'relu_block.3': 1})
    pipeline_net = AutoParallel(net_with_loss, parallel_mode="semi_auto")
    pipeline_net.pipeline(stages=2)
    ckpt_file = f'./zero_bubble/change_pipe/rank_{rank_id}/pipeline_changed{rank_id}.ckpt'
    infer_pipe = predict_each_rank(pipeline_net, ckpt_file, predict_data)

    # ===== Phase 7: Inference with transformed parallel checkpoints =====
    net_parallel = StageNet(weight_list=weight_list, micro_size=4, strategy1=((2, 1), (1, 2)),
                            strategy2=((2, 1), (1, 2)))
    parallel_net = AutoParallel(net_parallel, parallel_mode="semi_auto")
    ckpt_file = f'./zero_bubble/change/rank_{rank_id}/parallel_changed{rank_id}.ckpt'
    infer_parallel = predict_each_rank(parallel_net, ckpt_file, predict_data)

    # ===== Phase 8: Verify inference consistency =====
    if rank_id >= 4:
        allclose_nparray(infer_pipe.asnumpy(), infer_parallel.asnumpy(), 0.005, 0.005)
