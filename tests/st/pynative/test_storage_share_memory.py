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
""" test storage share memory """

import os
import pytest
import gc
import numpy as np

import mindspore as ms
from mindspore import Tensor
import mindspore.multiprocessing as mp
from mindspore.multiprocessing.reductions import reduce_storage
from tests.mark_utils import arg_mark


HAS_SHM_FILES = os.path.isdir("/dev/shm")
MAX_WAITING_TIME_IN_SECONDS = 30
# don't modify flag_tensor and target_tensor, used to assert
flag_tensor = Tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=ms.int32)
target_tensor = Tensor([[3,4,5], [6,7,8], [9,10,11]], dtype=ms.int32)


class leak_checker:
    def __init__(self):
        self.checked_pids = [os.getpid()]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if args[0] is None:
            assert not self.has_shm_files()
        return False

    def check_pid(self, pid):
        self.checked_pids.append(pid)

    def has_shm_files(self):
        if not HAS_SHM_FILES:
            return False
        result = self._has_shm_files()
        return result

    def _has_shm_files(self):
        gc.collect()
        names = ["mindspore_" + str(pid) for pid in self.checked_pids]
        for filename in os.listdir("/dev/shm"):
            for name in names:
                if filename.startswith(name):
                    return True
        return False


def copy_and_set_event(queue, event):
    t = Tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=ms.int32)
    t.copy_(target_tensor)
    queue.put(t)
    event.wait()

def simple_copy(tensor):
    tensor.copy_(target_tensor)
    return tensor

def send_tensor(queue, event):
    tmp = Tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=ms.int32)
    queue.put(tmp)
    queue.put(tmp)
    event.wait()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ipc_share_memory_single_put_and_get(mode):
    """
    Feature: share tensor memory.
    Description: test sharing tensor between different process.
    Expectation: successfully share tensor.
    """
    ms.context.set_context(mode=mode)
    ctx = mp.get_context('spawn')

    def do_test():
        q = ctx.Queue()
        e = ctx.Event()

        p = ctx.Process(target=copy_and_set_event, args=(q,e))
        p.daemon = True
        lc.check_pid(p.pid)
        p.start()

        out = q.get()
        e.set()
        p.join(100)
        assert np.allclose(out, target_tensor, rtol=1e-5, equal_nan=True)
        assert not p.is_alive()

    with leak_checker() as lc:
        do_test()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ipc_share_memory_multi_put_and_get(mode):
    """
    Feature: share tensor memory.
    Description: test sharing tensor between different process,
                 and executing queue.get() and queue.put() multi times.
    Expectation: successfully share tensor.
    """
    ms.context.set_context(mode=mode)
    ctx = mp.get_context('spawn')

    def test_receive():
        q = ctx.Queue()
        e = ctx.Event()

        p = ctx.Process(target=send_tensor, args=(q, e))
        p.daemon = True
        lc.check_pid(p.pid)
        p.start()

        t1 = q.get()
        t2 = q.get()
        assert np.allclose(t1, flag_tensor, rtol=1e-5, equal_nan=True)
        assert np.allclose(t2, flag_tensor, rtol=1e-5, equal_nan=True)

        e.set()
        p.join(100)
        assert not p.is_alive()

    with leak_checker() as lc:
        test_receive()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ipc_share_list(mode):
    """
    Feature: share tensor memory.
    Description: test sharing a list between different process.
    Expectation: successfully share tensor.
    """
    ms.context.set_context(mode=mode)
    ctx = mp.get_context('spawn')

    def do_test():
        x = Tensor([[1,2,3], [4,5,6], [7,8,9]], dtype=ms.int32)
        data = [x.storage(), x]
        q = ctx.Queue()
        q.put(data)
        new_data = q.get(timeout=1)

        assert new_data[0].nbytes() == 36
        assert np.allclose(new_data[1], flag_tensor, rtol=1e-5, equal_nan=True)

    with leak_checker():
        do_test()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ipc_pool(mode):
    """
    Feature: share tensor memory.
    Description: test sharing tensor between different process by using mp.Pool.
    Expectation: successfully share tensor.
    """
    ms.context.set_context(mode=mode)
    ctx = mp.get_context('spawn')

    def do_test():
        p = ctx.Pool(2)
        for proc in p._pool:  # pylint:disable=protected-access
            lc.check_pid(proc.pid)

        t1 = Tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=ms.int32)
        t2 = Tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=ms.int32)
        buffers = [t1, t2]
        results = p.map(simple_copy, buffers, 1)
        p.close()
        p.join()

        assert len(results) == len(buffers)
        for r in results:
            res = np.allclose(r, target_tensor, rtol=1e-5, equal_nan=True)
            assert res

    with leak_checker() as lc:
        do_test()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ipc_ascend_tensor(mode):
    """
    Feature: share tensor memory.
    Description: don't support sharing Ascend Tensor between different processes.
    Expectation: throw exception.
    """
    ms.context.set_context(mode=mode)

    t = Tensor([[1,2,3], [4,5,6], [7,8,9]], dtype=ms.int32)
    t1 = t.move_to('Ascend')

    with pytest.raises(RuntimeError):
        reduce_storage(t1.untyped_storage())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ipc_tensor_is_shared(mode):
    """
    Feature: check whether tensor is in the shared memory.
    Description: check whether tensor is in the shared memory.
    Expectation: return true when tensor is in the shared memory, otherwise return false.
    """
    ms.context.set_context(mode=mode)
    ctx = mp.get_context('spawn')

    t = Tensor([[1,2,3], [4,5,6], [7,8,9]], dtype=ms.int32)
    assert not t.is_shared()
    q = ctx.Queue()
    q.put(t)
    out = q.get()
    assert t.is_shared()
    assert out.is_shared()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ipc_empty_tensor(mode):
    """
    Feature: share tensor memory.
    Description: share empty tensor between different processes.
    Expectation: successfully share tensor.
    """
    q = mp.Queue()
    empty = Tensor([], dtype=ms.int32)
    q.put(empty)
    out = q.get()
    assert out.nbytes == 0
    assert out.dtype == ms.int32

    input_size = (0,)
    assert np.allclose(out.shape, input_size)
