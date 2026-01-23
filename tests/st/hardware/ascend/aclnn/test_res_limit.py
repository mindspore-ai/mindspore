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
"""test res limit"""
import os
import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor, mint
from mindspore.nn import Cell
from tests.mark_utils import arg_mark

context.set_context(mode=context.PYNATIVE_MODE)


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.op = mint.sin

    def construct(self, x):
        out = self.op(x)
        return out

def get_hash_id(texts):
    import re
    matches = re.findall(r'hash id:(\d+)', texts)
    return matches

def hash_id_change(texts, nums):
    matches = get_hash_id(texts)
    if len(matches) < nums:
        return False
    seen = set()
    for hash_id in matches:
        if hash_id in seen:
            return False
        seen.add(hash_id)
    return True

def test_hash_with_device_res_limit():
    """
    Feature: test aclnn cache with different device resource limit
    Description: test aclnn cache with different device resource limit
    Expectation: hash id change
    """
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    device_id = int(os.getenv("DEVICE_ID"))
    mindspore.runtime.set_device_limit(device_id, 8, 8)
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    mindspore.runtime.set_device_limit(device_id, 8, 20)
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    mindspore.runtime.set_device_limit(device_id, 4, 20)
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_aclnn_hash_with_device_res_limit():
    """
    Feature: test aclnn cache with different device resource limit
    Description: test aclnn cache with different device resource limit
    Expectation: hash id change
    """
    os.environ["VLOG_v"] = "20002"
    ret = os.system("pytest -sv test_res_limit.py::test_hash_with_device_res_limit > log_hash_id_1.txt 2>&1")
    assert not ret
    assert os.path.exists("log_hash_id_1.txt")
    ret = os.popen("grep -i 'miss cache, with hash id:' log_hash_id_1.txt").read()
    assert ret
    assert hash_id_change(ret, 4)
    os.system("rm -rf log_hash_id_1.txt")
    del os.environ["VLOG_v"]


def test_hash_without_res_limit():
    """
    Feature: test aclnn cache without different device resource limit
    Description: test aclnn cache without different device resource limit
    Expectation: hash id not change
    """
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    device_id = int(os.getenv("DEVICE_ID"))
    mindspore.runtime.set_device_limit(device_id, 8, 8)
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_aclnn_hash_without_res_limit():
    """
    Feature: test aclnn cache without different device resource limit
    Description: test aclnn cache without different device resource limit
    Expectation: hash id not change
    """
    os.environ["VLOG_v"] = "20002"
    ret = os.system("pytest -sv test_res_limit.py::test_hash_without_res_limit > log_hash_id_2.txt 2>&1")
    assert not ret
    assert os.path.exists("log_hash_id_2.txt")
    ret = os.popen("grep -i 'miss cache, with hash id:' log_hash_id_2.txt").read()
    assert ret
    hash_ids = get_hash_id(ret)
    assert len(hash_ids) == 2
    assert hash_ids[0] != hash_ids[1]
    os.system("rm -rf log_hash_id_2.txt")
    del os.environ["VLOG_v"]


def test_hash_with_stream_res_limit1():
    """
    Feature: test aclnn cache with different stream resource limit
    Description: test aclnn cache with different stream resource limit
    Expectation: hash id change
    """
    s1 = mindspore.runtime.current_stream()
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    mindspore.runtime.set_stream_limit(s1, 4, 8)
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    mindspore.runtime.set_stream_limit(s1, 8, 8)
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)

    mindspore.runtime.set_stream_limit(s1, 4, 16)
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    net = Net()
    net(x)


def test_hash_with_stream_res_limit2():
    """
    Feature: test aclnn cache with different stream resource limit
    Description: test aclnn cache with different stream resource limit
    Expectation: hash id change
    """
    s1 = mindspore.runtime.Stream()
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    with mindspore.runtime.StreamCtx(s1):
        mint.sin(x)

    mindspore.runtime.set_stream_limit(s1, 4, 8)
    with mindspore.runtime.StreamCtx(s1):
        mint.sin(x)

    mindspore.runtime.set_stream_limit(s1, 8, 8)
    with mindspore.runtime.StreamCtx(s1):
        mint.sin(x)

    mindspore.runtime.set_stream_limit(s1, 4, 16)
    with mindspore.runtime.StreamCtx(s1):
        mint.sin(x)

def test_hash_with_stream_res_limit3():
    """
    Feature: test aclnn cache with different stream resource limit
    Description: test aclnn cache with different stream resource limit
    Expectation: hash id change
    """
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    s1 = mindspore.runtime.Stream()
    mindspore.runtime.set_stream_limit(s1, 4, 8)
    with mindspore.runtime.StreamCtx(s1):
        mint.sin(x)

    s2 = mindspore.runtime.Stream()
    mindspore.runtime.set_stream_limit(s2, 2, 8)
    with mindspore.runtime.StreamCtx(s2):
        mint.sin(x)

    s3 = mindspore.runtime.Stream()
    mindspore.runtime.set_stream_limit(s3, 8, 8)
    with mindspore.runtime.StreamCtx(s3):
        mint.sin(x)

    s4 = mindspore.runtime.Stream()
    mindspore.runtime.set_stream_limit(s4, 4, 16)
    with mindspore.runtime.StreamCtx(s4):
        mint.sin(x)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=8.00)
def test_aclnn_hash_with_stream_res_limit():
    """
    Feature: test aclnn cache with different stream resource limit
    Description: test aclnn cache with different stream resource limit
    Expectation: hash id change
    """
    os.environ["VLOG_v"] = "20002"
    ret = os.system("pytest -sv test_res_limit.py::test_hash_with_stream_res_limit1 > log_hash_id_3.txt 2>&1")
    assert not ret
    assert os.path.exists("log_hash_id_3.txt")
    ret = os.popen("grep -i 'miss cache, with hash id:' log_hash_id_3.txt").read()
    assert ret
    assert hash_id_change(ret, 4)
    os.system("rm -rf log_hash_id_3.txt")

    ret = os.system("pytest -sv test_res_limit.py::test_hash_with_stream_res_limit2 > log_hash_id_4.txt 2>&1")
    assert not ret
    assert os.path.exists("log_hash_id_4.txt")
    ret = os.popen("grep -i 'miss cache, with hash id:' log_hash_id_4.txt").read()
    assert ret
    assert hash_id_change(ret, 4)
    os.system("rm -rf log_hash_id_4.txt")

    ret = os.system("pytest -sv test_res_limit.py::test_hash_with_stream_res_limit3 > log_hash_id_5.txt 2>&1")
    assert not ret
    assert os.path.exists("log_hash_id_5.txt")
    ret = os.popen("grep -i 'miss cache, with hash id:' log_hash_id_5.txt").read()
    assert ret
    assert hash_id_change(ret, 4)
    os.system("rm -rf log_hash_id_5.txt")
    del os.environ["VLOG_v"]


def test_hash_without_stream_res_limit():
    """
    Feature: test aclnn cache without different stream resource limit
    Description: test aclnn cache without different stream resource limit
    Expectation: hash id not change
    """
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    s1 = mindspore.runtime.Stream()
    mindspore.runtime.set_stream_limit(s1, 4, 8)
    with mindspore.runtime.StreamCtx(s1):
        mint.sin(x)

    with mindspore.runtime.StreamCtx(s1):
        mint.sin(x)

    with mindspore.runtime.StreamCtx(s1):
        mint.sin(x)

    s2 = mindspore.runtime.Stream()
    mindspore.runtime.set_stream_limit(s2, 2, 8)
    with mindspore.runtime.StreamCtx(s2):
        mint.sin(x)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_aclnn_hash_without_stream_res_limit():
    """
    Feature: test aclnn cache with different stream resource limit
    Description: test aclnn cache with different stream resource limit
    Expectation: hash id change
    """
    os.environ["VLOG_v"] = "20002"
    ret = os.system("pytest -sv test_res_limit.py::test_hash_without_stream_res_limit > log_hash_id_6.txt 2>&1")
    assert not ret
    assert os.path.exists("log_hash_id_6.txt")
    ret = os.popen("grep -i 'miss cache, with hash id:' log_hash_id_6.txt").read()
    assert ret
    hash_ids = get_hash_id(ret)
    assert len(hash_ids) == 2
    assert hash_ids[0] != hash_ids[1]
    os.system("rm -rf log_hash_id_6.txt")
    del os.environ["VLOG_v"]
