# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test aclnn cache"""
import os
import time
import multiprocessing
import psutil
from tests.mark_utils import arg_mark

import numpy as np
import mindspore
from mindspore import mint, context, mutable, jit
from mindspore.nn import Cell


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.op1 = mint.sin
        self.op2 = mint.cos

    def construct(self, x):
        out = self.op1(x)
        out = self.op2(out)
        out = self.op1(out)
        out = self.op2(out)
        out = self.op1(out)
        out = self.op2(out)
        out = self.op1(out)
        return out

def test_kbyk_aclnn_cache_1():
    """
    Feature: test aclnn cache.
    Description: set global aclnn cache
    Expectation: set aclnn cache failed
    """
    x = mindspore.Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend", jit_level="O0")
    mindspore.device_context.ascend.op_tuning.aclnn_cache(True)
    net = Net()
    for _ in range(5):
        net(x)


def test_kbyk_aclnn_cache_2():
    """
    Feature: test aclnn cache.
    Description: set global aclnn cache
    Expectation: set aclnn cache failed
    """
    x = mindspore.Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend", jit_level="O0")
    mindspore.device_context.ascend.op_tuning.aclnn_cache(cache_queue_length=100)
    net = Net()
    for _ in range(5):
        net(x)


def test_pyboost_aclnn_cache():
    """
    Feature: test aclnn cache.
    Description: set global aclnn cache
    Expectation: set aclnn cache failed
    """
    x = mindspore.Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend")
    net = Net()
    for _ in range(5):
        net(x)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_aclnn_cache_length_kbyk():
    """
    Feature: aclnn cache
    Description: set aclnn cache length to 100
    Expectation: set aclnn cache length failed
    """
    os.environ["VLOG_v"] = "20002"
    os.system("pytest -sv test_aclnn_cache.py::test_kbyk_aclnn_cache_2 > log_cache_kbyk_1.txt 2>&1")
    assert os.path.exists("log_cache_kbyk_1.txt")
    ret = os.system("grep -i 'Set aclnn cache queue length of kbyk to 100' log_cache_kbyk_1.txt")
    assert ret == 0
    os.system("rm -rf log_cache_kbyk_1.txt")
    del os.environ["VLOG_v"]

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_global_aclnn_cache_kbyk():
    """
    Feature: global aclnn cache
    Description: set global aclnn cache
    Expectation: set global aclnn cache failed
    """
    # use ms global aclnn cache
    os.environ["VLOG_v"] = "20002"
    os.system("pytest -sv test_aclnn_cache.py::test_kbyk_aclnn_cache_1 > log_cache_kbyk_2.txt 2>&1")
    assert os.path.exists("log_cache_kbyk_2.txt")
    ret_miss = os.popen("grep -i ' gen executor miss cache, hash id: ' log_cache_kbyk_2.txt | wc -l").read()
    assert int(ret_miss.strip()) == 2
    os.system("rm -rf log_cache_kbyk_2.txt")
    del os.environ["VLOG_v"]

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_global_aclnn_cache_pyboost():
    """
    Feature: global aclnn cache
    Description: set global aclnn cache
    Expectation: set global aclnn cache failed
    """
    # use ms global aclnn cache
    os.environ["VLOG_v"] = "20002"
    os.system("pytest -sv test_aclnn_cache.py::test_pyboost_aclnn_cache > log_cache_pyboost.txt 2>&1")
    ret_miss = os.popen("grep -i 'miss cache, with hash id:' log_cache_pyboost.txt | wc -l").read()
    assert int(ret_miss.strip()) == 2
    os.system("rm -rf log_cache_pyboost.txt")
    del os.environ["VLOG_v"]

def test_aclnn_cache_with_multi_input_1():
    """
    Feature: aclnn cache with 400 input
    Description: aclnn cache with 400 input
    Expectation: aclnn cache is failed
    """
    x_list = []
    for _ in range(400):
        x = np.random.uniform(-1, 1, (1000))
        x_list.append(mindspore.Tensor(x, mindspore.float32))
    x_list = tuple(x_list)
    for _ in range(5):
        mint.cat([x.view(-1).contiguous() for x in x_list], dim=0)

def test_aclnn_cache_with_multi_input_2():
    """
    Feature: aclnn cache with 1000 input
    Description: aclnn cache with 1000 input
    Expectation: aclnn cache is failed
    """
    x_list = []
    for _ in range(1000):
        x = np.random.uniform(-1, 1, (1000))
        x_list.append(mindspore.Tensor(x, mindspore.float32))
    x_list = tuple(x_list)
    for _ in range(5):
        mint.cat([x.view(-1).contiguous() for x in x_list], dim=0)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=12.00)
def test_aclnn_cache_with_multi_input():
    """
    Feature: aclnn cache with multi input
    Description: aclnn cache with multi input
    Expectation: aclnn cache is failed
    """
    # 400 inputs is supported, hash id is not 0 and cache is available
    os.system("pytest -sv test_aclnn_cache.py::test_aclnn_cache_with_multi_input_1 > log_cache_multi_input_1.txt 2>&1")
    ret_miss = os.popen("grep -i 'aclnnCat cache is available, but hash id is 0, do not use cache.' \
                         log_cache_multi_input_1.txt | wc -l").read()
    assert int(ret_miss.strip()) == 0
    os.system("rm -rf log_cache_multi_input_1.txt")

    # 1000 inputs is not supported, cache is available but hash id is 0
    os.system("pytest -sv test_aclnn_cache.py::test_aclnn_cache_with_multi_input_2 > log_cache_multi_input_2.txt 2>&1")
    ret_miss = os.popen("grep -i 'aclnnCat cache is available, but hash id is 0, do not use cache.' \
                         log_cache_multi_input_2.txt | wc -l").read()
    assert int(ret_miss.strip()) == 5
    os.system("rm -rf log_cache_multi_input_2.txt")


class SimpleNet1(Cell):
    def __init__(self):
        super().__init__()
        self.op1 = mint.sin
        self.op2 = mint.nonzero
        self.op3 = mint.cos

    def construct(self, x):
        out = x
        for _ in range(mutable(100)):
            x += x
            x = self.op2(x)
            out = self.op1(x)
            out = self.op3(out)
        return out

def extract_memory_values(text):
    return float(text.split("memory usage:")[1].split()[0])

def memory_monitor(pid, interval=0.1):
    # monitor the host memory usage of main process
    try:
        while True:
            mem = psutil.Process(pid).memory_info().rss / 1024 / 1024
            print(f"memory usage: {mem} MB", flush=True)
            time.sleep(interval)
    except psutil.NoSuchProcess:
        print("Memory monitor stopped.")

def test_aclnn_cache_memory_kbk():
    """
    Feature: aclnn cache host memory
    Description: test aclnn cache host memory
    Expectation: aclnn cache host memory leak
    """
    os.environ['MS_DEV_RUNTIME_CONF'] = "aclnn_cache_queue_length:1"
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend", jit_level="O0")
    net = SimpleNet1()
    pid = os.getpid()
    monitor_proc = multiprocessing.Process(target=memory_monitor, args=(pid, 15))
    monitor_proc.daemon = True
    monitor_proc.start()
    random_array = np.random.rand(10, 1500, 64)
    x = mindspore.Tensor(random_array.astype(np.float32))
    net(x)
    del os.environ["MS_DEV_RUNTIME_CONF"]

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=14108.00)
def test_aclnn_cache_host_memory_kbk():
    """
    Feature: aclnn cache host memory
    Description: test aclnn cache host memory
    Expectation: aclnn cache host memory leak
    """
    os.system("pytest -sv test_aclnn_cache.py::test_aclnn_cache_memory_kbk > log_cache_memory_kbk.txt 2>&1")
    memory_usage_last1 = os.popen("grep 'memory usage: ' log_cache_memory_kbk.txt | tail -n 1").read()
    memory_usage_last2 = os.popen("grep 'memory usage: ' log_cache_memory_kbk.txt | tail -n 2 | head -n 1").read()
    last1, last2 = extract_memory_values(memory_usage_last1), extract_memory_values(memory_usage_last2)
    diff_percent = abs(last1 - last2) / last1
    # the 900th step and 1000th step memory usage change should be less than 0.5% in kbk
    assert diff_percent <= 0.005
    os.system("rm -rf log_cache_memory_kbk.txt")

class SimpleNet2(Cell):
    def __init__(self):
        super().__init__()
        self.op = mint.cat

    def construct(self, x):
        out = self.op([x, x, x, x, x, x], dim=0)
        return out

def test_aclnn_cache_memory_pyboost():
    """
    Feature: aclnn cache host memory
    Description: test aclnn cache host memory
    Expectation: aclnn cache host memory leak
    """
    context.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend")
    net = SimpleNet2()
    for i in range(2000):
        dim1 = np.random.randint(1, 11)
        dim2 = np.random.randint(1000, 2000)
        dim3 = np.random.randint(20, 80)
        random_array = np.random.rand(dim1, dim2, dim3)
        x = mindspore.Tensor(random_array.astype(np.float32))
        net(x)
        if i % 200 == 0:
            print(f'step {i}, memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB')

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=38.00)
def test_aclnn_cache_host_memory_pyboost():
    """
    Feature: aclnn cache host memory
    Description: test aclnn cache host memory
    Expectation: aclnn cache host memory leak
    """
    os.environ["MS_DEV_RUNTIME_CONF"] = "aclnn_cache_queue_length:100"
    os.system("pytest -sv test_aclnn_cache.py::test_aclnn_cache_memory_pyboost > log_cache_memory_pyboost.txt 2>&1")
    memory_usage_last1 = os.popen("grep 'memory usage:' log_cache_memory_pyboost.txt | tail -n 1").read()
    memory_usage_last2 = os.popen("grep 'memory usage:' log_cache_memory_pyboost.txt | tail -n 2 | head -n 1").read()
    last1, last2 = extract_memory_values(memory_usage_last1), extract_memory_values(memory_usage_last2)
    diff_percent = abs(last1 - last2) / last1
    # the 900th step and 1000th step memory usage change should be less than 0.1% in pyboost
    assert diff_percent <= 0.001
    os.system("rm -rf log_cache_memory_pyboost.txt")
    del os.environ["MS_DEV_RUNTIME_CONF"]

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_sync_aclnn_op_kbyk():
    """
    Feature: sync aclnn op
    Description: test sync aclnn op
    Expectation: run successfully
    """
    os.environ["MS_DEV_RUNTIME_CONF"] = "aclnn_cache_queue_length:0"
    class NonZeroNet(Cell):
        @jit
        def construct(self, x):
            return mint.nonzero(x)
    net = NonZeroNet()

    # test static shape
    x = mindspore.Tensor(shape=[100, 1000], dtype=mindspore.float32)
    out = net(x)
    print(out)

    # test dynamic shape
    t = mindspore.Tensor(shape=[None, None], dtype=mindspore.float32)
    net.set_inputs(t)
    x = mindspore.Tensor(shape=[100, 1000], dtype=mindspore.float32)
    out = net(x)
    print(out)

    del os.environ["MS_DEV_RUNTIME_CONF"]

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_sync_aclnn_op_pyboost():
    """
    Feature: sync aclnn op
    Description: test sync aclnn op
    Expectation: run successfully
    """
    os.environ["MS_DEV_RUNTIME_CONF"] = "aclnn_cache_queue_length:0"
    class NonZeroNet(Cell):
        def construct(self, x):
            return mint.nonzero(x)
    net = NonZeroNet()

    # test static shape
    x = mindspore.Tensor(shape=[100, 1000], dtype=mindspore.float32)
    out = net(x)
    print(out)

    # test dynamic shape
    t = mindspore.Tensor(shape=[None, None], dtype=mindspore.float32)
    net.set_inputs(t)
    x = mindspore.Tensor(shape=[100, 1000], dtype=mindspore.float32)
    out = net(x)
    print(out)

    del os.environ["MS_DEV_RUNTIME_CONF"]
