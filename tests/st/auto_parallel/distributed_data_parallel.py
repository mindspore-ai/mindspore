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
"""Distributed Data Parallel Example"""
import time
import numpy as np
import mindspore.dataset as ds
import mindspore as ms
from mindspore import Parameter, Tensor, ops, nn
from mindspore.parallel.distributed import DistributedDataParallel
from mindspore.mint.optim import AdamW
from mindspore.mint.distributed.distributed import init_process_group, get_rank
from mindspore.communication import GlobalComm
from mindspore.graph.api import _pynative_executor


def get_data_parallel_group():
    return GlobalComm.WORLD_COMM_GROUP


class Network(nn.Cell):
    """Network"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.unused_cell0 = nn.Dense(10, 10, weight_init="normal", bias_init="zeros")
        self.head = nn.Dense(10, 10, weight_init="normal", bias_init="zeros")
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28 * 28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.Dense(512, 4096, weight_init="normal", bias_init="zeros"),
            nn.Dense(4096, 512, weight_init="normal", bias_init="zeros"),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros"),
        )
        self.skipped_cell1 = nn.Dense(10, 10, weight_init="normal", bias_init="zeros")
        self.skipped_cell1.weight.requires_grad = False
        self.skipped_cell1.bias.requires_grad = False

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        logits = self.head(logits)
        return logits


def generate_fake_dataset(batch_size, num_samples):
    np.random.seed(get_rank())
    data = np.random.rand(num_samples, 1, 28, 28).astype(np.float32)
    labels = np.random.randint(0, 10, size=(num_samples,)).astype(np.int32)
    return ds.NumpySlicesDataset((data, labels), shuffle=False).batch(batch_size)


class Accumulator:
    """
    Feature: grads accumulator
    Description: sum grad of each micro batch into one, which equals to grads of global batch.
    """

    def __init__(self, optimizer, accumulate_step, clip_norm=1.0):
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(
            prefix="accumulate_", init="zeros"
        )
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
        self.counter = Parameter(Tensor(1, ms.int32), "counter_")
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        self.map = ops.HyperMap()
        self.grad_reducer = nn.DistributedGradReducer(
            optimizer.parameters, mean=True, group=get_data_parallel_group()
        )

    def __call__(self, grads):
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            self.inner_grads = self.grad_reducer(self.inner_grads)
            self.optimizer(self.inner_grads)
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
        ops.assign_add(self.counter, Tensor(1, ms.int32))
        return True


def train_step_full_batch(net, data_set, grad_fn, optimizer, enable_ddp_flag):
    """
    Description: global batch DDP test case
    """
    loss = []
    if not enable_ddp_flag:
        grad_reducer = nn.DistributedGradReducer(
            optimizer.parameters, mean=True, group=get_data_parallel_group()
        )

    for epoch in range(2):
        i = 0
        for image, label in data_set:
            start_time = time.time()
            (loss_value, _), grads = grad_fn(image, label)
            if not enable_ddp_flag:
                grads = grad_reducer(grads)
            optimizer(grads)
            loss.append(loss_value.asnumpy())
            end_time = time.time()
            print(
                "epoch: %s, step: %s, loss is %.15f, time is %.2f ms rank %s"
                % (epoch, i, loss_value, (end_time - start_time) * 1000, get_rank())
            )
            i += 1
            if enable_ddp_flag:
                net.zero_grad()
    return loss


def train_step_dgr_accumulate(data_set, grad_fn, accumulator):
    """
    Description: data parallel with grad accumulation case
    """
    for epoch in range(2):
        loss = []
        i = 0
        for image, label in data_set:
            start_time = time.time()
            (loss_value, _), grads = grad_fn(image, label)
            accumulator(grads)
            loss.append(loss_value.asnumpy())
            end_time = time.time()
            print(
                "epoch: %s, step: %s, loss is %.15f, time is %.2f ms, from %s"
                % (epoch, i, loss_value, (end_time - start_time) * 1000, get_rank())
            )
            i += 1
    return loss


def train_step_ddp_accumulate(net, data_set, grad_fn, optimizer, accu_steps):
    """
    Description: DDP data parallel with grad accumulation case
    """
    accu_count = 1
    for epoch in range(2):
        i = 0
        loss = []
        for image, label in data_set:
            start_time = time.time()
            if accu_count < accu_steps:
                with net.no_sync():
                    (loss_value, _), grads = grad_fn(image, label)
                    accu_count += 1
            else:
                (loss_value, _), grads = grad_fn(image, label)
                optimizer(grads)
                if isinstance(net, DistributedDataParallel):
                    net.zero_grad()
                accu_count = 1
            loss.append(loss_value.asnumpy())
            end_time = time.time()
            print(
                "epoch: %s, step: %s, loss is %.15f, time is %.2f ms rank %s"
                % (epoch, i, loss_value, (end_time - start_time) * 1000, get_rank())
            )
            i += 1
    return loss


def test_full_batch_DDP_without_bucket_rebuilt(reducer_mode="PythonReducer"):
    """
    Description: DDP data parallel without rebuilt bucket
    find unused params; but do not rebuild bucket
    """
    ms.set_context(
        device_target="Ascend",
        mode=ms.PYNATIVE_MODE,
        pynative_synchronize=True,
        deterministic="ON",
    )
    init_process_group()
    data_set = generate_fake_dataset(batch_size=32, num_samples=320)
    ms.set_seed(0)
    net = Network()

    def forward_fn(data, target):
        """forward propagation"""
        logits = net(data)
        loss = loss_fn(logits, target)
        return loss, logits

    enable_ddp_flag = False
    optimizer = AdamW(net.trainable_params(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()
    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    loss_dgr = train_step_full_batch(net, data_set, grad_fn, optimizer, enable_ddp_flag)

    enable_ddp_flag = True
    data_set = generate_fake_dataset(batch_size=32, num_samples=320)
    ms.set_seed(get_rank())
    net = Network()
    net.recompute()
    net = DistributedDataParallel(
        module=net, bucket_cap_mb=None, average_in_collective=True, static_graph=False,
        find_unused_parameters=True, reducer_mode=reducer_mode
    )

    optimizer = AdamW(net.trainable_params(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()
    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    loss_ddp = train_step_full_batch(net, data_set, grad_fn, optimizer, enable_ddp_flag)

    assert np.allclose(loss_ddp, loss_dgr, 1e-12, 1e-12)


def test_full_batch_DDP_with_bucket_rebuilt(reducer_mode="PythonReducer"):
    """
    Description: DDP data parallel with rebuilt bucket
    rebuild bucket; but do not find unused params
    """
    ms.set_context(
        device_target="Ascend",
        mode=ms.PYNATIVE_MODE,
        pynative_synchronize=True,
        deterministic="ON",
    )
    init_process_group()
    data_set = generate_fake_dataset(batch_size=32, num_samples=320)
    ms.set_seed(0)
    net = Network()

    def forward_fn(data, target):
        """forward propagation"""
        logits = net(data)
        loss = loss_fn(logits, target)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    optimizer = AdamW(net.trainable_params(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()

    enable_ddp_flag = False
    loss_dgr = train_step_full_batch(net, data_set, grad_fn, optimizer, enable_ddp_flag)

    enable_ddp_flag = True
    data_set = generate_fake_dataset(batch_size=32, num_samples=320)
    ms.set_seed(get_rank())
    net = Network()
    net.recompute()
    net = DistributedDataParallel(
        module=net, bucket_cap_mb=None, average_in_collective=True, static_graph=True,
        find_unused_parameters=False, reducer_mode=reducer_mode
    )
    optimizer = AdamW(net.trainable_params(), 1e-4)
    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    loss_fn = nn.CrossEntropyLoss()
    loss_ddp = train_step_full_batch(net, data_set, grad_fn, optimizer, enable_ddp_flag)

    assert np.allclose(loss_ddp, loss_dgr, 1e-12, 1e-12)


def test_accumulate_batch_DDP_with_bucket_rebuilt(reducer_mode="PythonReducer"):
    """
    Description: DDP data parallel with rebuilt bucket
    """
    ms.set_context(
        device_target="Ascend",
        mode=ms.PYNATIVE_MODE,
        pynative_synchronize=True,
        deterministic="ON",
    )
    init_process_group()
    accu_steps = 4
    data_set = generate_fake_dataset(batch_size=32, num_samples=320)
    ms.set_seed(get_rank())
    net = Network()
    net.recompute()
    net = DistributedDataParallel(
        module=net, bucket_cap_mb=None, average_in_collective=True, static_graph=True,
        find_unused_parameters=False, reducer_mode=reducer_mode
    )
    optimizer = AdamW(net.trainable_params(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def forward_fn(data, target):
        """forward propagation"""
        logits = net(data)
        loss = loss_fn(logits, target) / accu_steps
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    loss_ddp = train_step_ddp_accumulate(net, data_set, grad_fn, optimizer, accu_steps)

    accu_steps = 4
    data_set = generate_fake_dataset(batch_size=32, num_samples=320)
    ms.set_seed(0)
    net = Network()
    optimizer = AdamW(net.trainable_params(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()
    accumulator = Accumulator(optimizer, accu_steps)
    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    loss_dgr = train_step_dgr_accumulate(data_set, grad_fn, accumulator)

    assert np.allclose(loss_ddp, loss_dgr, 1e-12, 1e-12)


def test_accumulate_batch_DDP_without_bucket_rebuilt(reducer_mode="PythonReducer"):
    """
    Description: DDP data parallel without rebuilt bucket
    """
    ms.set_context(
        device_target="Ascend",
        mode=ms.PYNATIVE_MODE,
        pynative_synchronize=True,
        deterministic="ON",
    )
    init_process_group()
    accu_steps = 4
    data_set = generate_fake_dataset(batch_size=32, num_samples=320)
    ms.set_seed(get_rank())
    net = Network()
    net.recompute()
    net = DistributedDataParallel(
        module=net, bucket_cap_mb=None, average_in_collective=True, static_graph=False,
        find_unused_parameters=True, reducer_mode=reducer_mode
    )
    optimizer = AdamW(net.trainable_params(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()
    accumulator = Accumulator(optimizer, accu_steps)

    def forward_fn(data, target):
        """forward propagation"""
        logits = net(data)
        loss = loss_fn(logits, target) / accu_steps
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    loss_ddp = train_step_ddp_accumulate(net, data_set, grad_fn, optimizer, accu_steps)

    accu_steps = 4
    data_set = generate_fake_dataset(batch_size=32, num_samples=320)
    ms.set_seed(0)
    net = Network()
    optimizer = AdamW(net.trainable_params(), 1e-4)
    loss_fn = nn.CrossEntropyLoss()
    accumulator = Accumulator(optimizer, accu_steps)
    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    loss_dgr = train_step_dgr_accumulate(data_set, grad_fn, accumulator)

    assert np.allclose(loss_ddp, loss_dgr, 1e-12, 1e-12)


def test_cpp_reducer_bucket_memory_allocated():
    """
    Description: CppReducer bucket tensors created from spec have usable device memory.
    """
    ms.set_context(
        device_target="Ascend",
        mode=ms.PYNATIVE_MODE,
        pynative_synchronize=True,
        deterministic="ON",
    )
    init_process_group()
    ms.set_seed(get_rank())
    net = DistributedDataParallel(
        module=Network(),
        init_sync=False,
        bucket_cap_mb=None,
        average_in_collective=True,
        static_graph=False,
        find_unused_parameters=True,
        reducer_mode="CppReducer",
    )

    buckets = net.reducer.get_bucket_for_debug()
    assert len(buckets) > 1
    net.zero_grad()
    _pynative_executor.sync()
    for bucket in buckets:
        bucket_data = bucket.asnumpy()
        assert bucket_data.size > 0
        assert np.allclose(bucket_data, np.zeros_like(bucket_data))


def test_full_batch_DDP_with_bucket_rebuilt_cpp():
    test_full_batch_DDP_with_bucket_rebuilt(reducer_mode="CppReducer")


def test_full_batch_DDP_without_bucket_rebuilt_cpp():
    test_full_batch_DDP_without_bucket_rebuilt(reducer_mode="CppReducer")


def test_accumulate_batch_DDP_with_bucket_rebuilt_cpp():
    test_accumulate_batch_DDP_with_bucket_rebuilt(reducer_mode="CppReducer")


def test_accumulate_batch_DDP_without_bucket_rebuilt_cpp():
    test_accumulate_batch_DDP_without_bucket_rebuilt(reducer_mode="CppReducer")
