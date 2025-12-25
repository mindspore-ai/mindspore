# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""
test memory feature in ascend
"""
import re
import multiprocessing
import numpy as np
import shutil
import os
import acl
import sys

from mindspore import Tensor
from mindspore import context
import mindspore as ms
from mindspore import nn, jit, Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal

from tests.mark_utils import arg_mark
from tests.device_utils import set_device

sys.path.append("..")
from test_hal_util import run_cmd

GB_TO_BYTE = 1024 << 20
FLOAT32_SIZE = 4


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.ops = P.Abs()

    def construct(self, x):
        return self.ops(x)

class Network(nn.Cell):
    """Simple network

    Args:
        weight_shape: shape of weight

    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet((10, 10))
    """
    def __init__(self, weight_shape, tensor_size_gb=1):
        super().__init__()
        self.relu = P.ReLU()
        self.mul1 = P.Mul()
        self.weight_1 = Parameter(Tensor(np.ones(weight_shape), dtype=ms.float32), name="weight_1")
        self.tensor_size = int(1024 / 4 * tensor_size_gb)

    def construct(self, x):
        x = self.mul1(x, self.weight_1)
        a = Tensor(np.ones((self.tensor_size, 1024, 1024)), dtype=ms.float32)
        b = a + 1
        print(b)
        z = self.relu(x)
        return z


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_cache_vmm():
    """
    Feature: runtime memory api.
    Description: Test runtime memory empty cache api.
    Expectation: runtime.empty_cache api performs as expected.
    """
    set_device()
    os.environ['MS_ALLOC_CONF'] = "enable_vmm:true"

    net = Net()
    net(Tensor(2.0))
    reserved_size = ms.runtime.memory_reserved()
    assert reserved_size > 0
    ms.runtime.empty_cache()
    reserved_size = ms.runtime.memory_reserved()
    assert reserved_size == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_cache_without_vmm():
    """
    Feature: runtime memory api.
    Description: Test runtime memory empty cache api.
    Expectation: runtime.empty_cache api performs as expected.
    """
    set_device()
    os.environ['MS_ALLOC_CONF'] = "enable_vmm:false"
    for _ in range(1000):
        net = Net()
        net(Tensor(2.0))
        reserved_size = ms.runtime.memory_reserved()
        assert reserved_size > 0
        ms.runtime.empty_cache()
        reserved_size = ms.runtime.memory_reserved()
        assert reserved_size == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_cache_dryrun():
    """
    Feature: runtime memory api.
    Description: Test runtime memory empty cache api.
    Expectation: runtime.empty_cache api performs as expected.
    """
    set_device()
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "1"
    os.environ["RANK_ID"] = "0"

    net = Net()
    net(Tensor(2.0))
    reserved_size = ms.runtime.memory_reserved()
    assert reserved_size > 0
    ms.runtime.empty_cache()
    reserved_size = ms.runtime.memory_reserved()
    assert reserved_size == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_memory_replay():
    """
    Feature: runtime memory api.
    Description: Test runtime memory replay api.
    Expectation: success.
    """
    mem_tracker_path = "test_replay_mem_tracker"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    tracker_path = os.path.join(cur_dir, mem_tracker_path)
    try:
        model_log = tracker_path + "/model.log"
        if os.path.isdir(tracker_path):
            shutil.rmtree(tracker_path)
        os.makedirs(tracker_path)
        os.environ['MS_ALLOC_CONF'] = "enable_vmm:false"
        os.environ['MS_DEV_RUNTIME_CONF'] = "memory_statistics:True"
        cmd = f"python hal_dryrun_case.py &> {model_log}"
        ret = os.system(cmd)
        assert ret == 0
        model_memory_usage = extract_memory_usage(model_log)
        assert model_memory_usage is not None

        replay_log = tracker_path + "/replay.log"
        os.system('python -c "import mindspore as ms;'
                  f'ms.runtime.memory_replay(\'{os.path.join(tracker_path, "memory_block.csv")}\')" &> {replay_log}')
        assert ret == 0
        replay_memory_usage = extract_memory_usage(replay_log)
        assert replay_memory_usage is not None
        assert model_memory_usage == replay_memory_usage
    except Exception as e:
        remove_dir = ["kernel_meta", "offload", tracker_path]
        for d in remove_dir:
            if os.path.isdir(d):
                shutil.rmtree(d)
        raise e

def extract_memory_usage(filename):
    pattern = r'Actual peak memory usage \(with fragments\): (\d+)M'
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                memory_usage = int(match.group(1))
                return memory_usage
        return None

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_huge_page_reserve_vmm():
    """
    Feature: reserve huge page for vmm.
    Description: Test whether huge page memory is reserved for vmm.
    Expectation: When huge page is reserved, use normal memory.
    """
    device_id = int(os.getenv("DEVICE_ID", "0"))
    acl.rt.set_device(device_id)
    mem_free, _, ret = acl.rt.get_mem_info(4)
    huge_page_free_gb = round(mem_free / 1024 / 1024 / 1024, 2)
    reserve_mem = max((huge_page_free_gb - 1) * 0.6, 0.5)
    ms.runtime.set_memory(huge_page_reserve_size=f"{reserve_mem}GB")
    net_1 = Network((1024, 1024))
    input_tensor = Tensor(np.random.randn(1024, 1024).astype(np.float32))
    net_1(input_tensor)
    mem_free_after, _, ret_after = acl.rt.get_mem_info(4)
    mem_free_after_gb = round(mem_free_after / 1024 / 1024 / 1024, 2)
    assert mem_free_after_gb >= reserve_mem
    assert ret == 0 and ret_after == 0

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_huge_page_reserve_with_size_0():
    """
    Feature: reserve huge page for vmm.
    Description: Test when huge page size is 0.
    Expectation: When huge page is reserved, use normal memory.
    """
    device_id = int(os.getenv("DEVICE_ID", "0"))
    acl.rt.set_device(device_id)
    _, _, ret = acl.rt.get_mem_info(4)
    ms.runtime.set_memory(huge_page_reserve_size="0GB")
    net_1 = Network((1024, 1024))
    input_tensor = Tensor(np.random.randn(1024, 1024).astype(np.float32))
    net_1(input_tensor)
    mem_free_after, _, ret_after = acl.rt.get_mem_info(4)
    assert mem_free_after >= 0
    assert ret == 0 and ret_after == 0

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_small_allocator():
    """
    Feature: small allocator.
    Description: Test whether the fragment size of enabling the small pool is
                 less than that of disabling the small pool/
    Expectation: the fragment size of enabling the small pool is smaller.
    """
    os.environ["MS_DEV_HOST_BLOCKING_RUN"] = "1"
    os.environ["MS_DEV_LAUNCH_BLOCKING"] = "1"

    cwd = os.path.dirname(os.path.realpath(__file__))
    test_script_relpath = "test_small_allocator/test_small_allocator_script.py"
    test_script_abspath = os.path.join(cwd, test_script_relpath)
    _, stdout, _ = run_cmd(f"python {test_script_abspath}")
    fragment_without_small_pool = int(stdout.split()[-1])

    os.environ['MS_ALLOC_CONF'] = "enable_small_pool:true"
    _, stdout, _ = run_cmd(f"python {test_script_abspath}")
    fragment_with_small_pool = int(stdout.split()[-1])

    assert fragment_with_small_pool < fragment_without_small_pool

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_runtime_memory_stats_idle():
    """
    Feature: runtime memory api.
    Description: Test runtime.memory_stats api.
    Expectation: runtime.memory_stats api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    input1 = ms.Tensor(np.random.random([2, 2]), ms.float32)
    input2 = ms.Tensor(np.random.random([2, 2]), ms.float32)
    add = ms.ops.Add()
    output = add(input1, input2)
    del output
    del input1
    del input2

    # 512b for each input and output, so the total memory usage is 512 * 3
    assert ms.runtime.memory_stats()["total_idle_memory"] == (512 * 3)



def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)

# context.set_context(save_graphs = True, save_graphs_path="./grahp_jit")
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet(num_class=10)
    """

    def __init__(self, num_class=10):
        super().__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

    @jit
    def construct(self, x):
        """build network"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_model():
    net = LeNet5()
    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    for _ in range(10):
        net(input_tensor)
    return True

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_runtime_vmm_close_multi_model_train():
    """
    Feature: basic test for multi model train.
    Description: close vmm to train multi model.
    Expectation: train multi model successfully.
    """
    set_device()

    os.environ['MS_ALLOC_CONF'] = 'enable_vmm:False'
    ms.runtime.set_memory(max_size="10GB", increase_size="1GB")
    res = []
    with multiprocessing.Pool(processes=2) as pool:
        for _ in range(2):
            res.append(pool.apply_async(train_model))
        pool.close()
        pool.join()

        for i in res:
            assert i.get()
    del os.environ['MS_ALLOC_CONF']

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_enable_mem_huge_1g():
    """
    Feature: Use 1G huge page size when environment variable enable_mem_huge_1g is enabled.
    Description: Test whether 1G huge page is used when enable_mem_huge_1g is enabled.
    Expectation: The total reserved memory equals 1GB after enabling 1G huge page allocation.
    """
    os.environ["MS_ALLOC_CONF"] = "enable_mem_huge_1g:true"
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    input1 = ms.Tensor(np.random.random([1, 2]), ms.float32)
    transpose = ms.ops.Transpose()
    transpose(input1, (1, 0))

    assert ms.runtime.memory_stats()["total_reserved_memory"] == 1024 * 1024 * 1024


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_memory_stats_with_stream():
    """
    Feature: test memory stats with stream.
    Description: test memory stats with stream.
    Expectation: memory stats is correct with stream ctx.
    """
    set_device()
    s1 = ms.runtime.Stream()

    @ms.jit
    def test_func(x):
        x = x + 1
        with ms.runtime.StreamCtx(s1):
            out = x * 1
            out = out + 2
        return out
    out = test_func(ms.Tensor(shape=(1024, 1024), dtype=ms.float32))
    print(out)
    # only the 'out' is still in use, so the memory usage is 1024 * 1024 * 4
    # if the 'total_allocated_memory' is 1024 * 1024 * 4 * 2, it means the 'x' is not free
    assert int(ms.runtime.memory_stats()["total_allocated_memory"] / (1024 * 1024 * 4)) == 1
