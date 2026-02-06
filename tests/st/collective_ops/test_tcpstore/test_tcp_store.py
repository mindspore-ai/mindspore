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
import pytest
from datetime import timedelta
from mindspore import context
from mindspore.mint.distributed.distributed import (
    init_process_group,
    barrier,
    get_rank,
    TCPStore,
    get_world_size
)
#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True pytest -sv --disable-warnings  test_tcp_store.py
init_process_group()
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
this_rank = get_rank()
size = get_world_size()
start_port = 12668
if size % 2 != 0:
    raise RuntimeError("Group size should be divided by 2 exactly.")
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


def log_function_entry_exit(func):
    """
    Feature: log function entry exit
    Description: add log for func
    Expectation: success
    """
    def wrapper(*args, **kwargs):
        # 打印进入函数的信息
        print(f"Entering comm function: {func.__name__}", flush=True)
        # 调用原函数
        result = func(*args, **kwargs)
        # 打印退出函数的信息
        print(f"Exiting comm function: {func.__name__}", flush=True)
        return result
    return wrapper


@log_function_entry_exit
def test_TCPStore():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        TCPStore("127.0.0.1", start_port, is_master=True)
    with pytest.raises(TypeError):
        TCPStore("11")
    with pytest.raises(TypeError):
        TCPStore("127.0.0.1", "1234")
    with pytest.raises(TypeError):
        TCPStore("127.0.0.1", start_port, is_master="xx")
    with pytest.raises(TypeError):
        TCPStore("127.0.0.1", start_port, is_master=True, timeout=0)
    with pytest.raises(TypeError):
        TCPStore(0, start_port, is_master=True)
    with pytest.raises(TypeError):
        TCPStore("127.0.0.1", start_port, is_master=True, world_size="xx")
    with pytest.raises(TypeError):
        TCPStore("127.0.0.1", start_port, is_master=True, wait_for_workers="xx")
    barrier()


@log_function_entry_exit
def test_TCPStore_TypeError():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+1, is_master=True, timeout=timedelta(seconds=10))
        with pytest.raises(TypeError):
            store.set("key1", 1)
        with pytest.raises(TypeError):
            store.set(2, "{'a':1}")
        with pytest.raises(TypeError):
            store.set("key3", [1, 2, 3])
        with pytest.raises(TypeError):
            store.set("key5")
        with pytest.raises(TypeError):
            store.delete_key(4)
        with pytest.raises(TypeError):
            store.get(2)
    barrier()


@log_function_entry_exit
def test_set():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        store_master = TCPStore("127.0.0.1", start_port+2, is_master=True, timeout=timedelta(seconds=10))
        store_master.set("first_key", "value1")
    barrier()
    store = TCPStore("127.0.0.1", start_port+2, is_master=False, timeout=timedelta(seconds=10))
    if this_rank == 1:
        store.set("first_key", "value2")
    barrier()
    data = store.get("first_key")
    assert data.decode() == "value2"
    barrier()


@log_function_entry_exit
def test_get():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+3, is_master=True, timeout=timedelta(seconds=10))
        with pytest.raises(RuntimeError):
            store.get("second_key")
        store.set("second_key", "value")
    barrier()
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+3, is_master=False, timeout=timedelta(seconds=10))
        data = store.get("second_key")
        assert data.decode() == "value"
    barrier()


@log_function_entry_exit
def test_get_1G():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 1:
        store_master = TCPStore("127.0.0.1", start_port+4, is_master=True, timeout=timedelta(seconds=300))
        value = 'A' * 1024 * 1024 * 1024
        store_master.set("third_key", value)
    barrier()
    store = TCPStore("127.0.0.1", start_port+4, is_master=False, timeout=timedelta(seconds=300))
    data = store.get("third_key")
    assert len(data) == 1024 * 1024 * 1024
    assert data[0] == 65
    barrier()


@log_function_entry_exit
def test_delete():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+5, is_master=True, timeout=timedelta(seconds=10))
        store.set("first_key", "value")
        ret = store.delete_key("first_key")
        assert ret is True
        ret = store.delete_key("first_key")
        assert ret is False
        with pytest.raises(RuntimeError):
            store.get("first_key")
    barrier()
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+5, is_master=False, timeout=timedelta(seconds=10))
        with pytest.raises(RuntimeError):
            store.get("first_key")
    barrier()


@log_function_entry_exit
def test_ip_port():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+6, is_master=True, timeout=timedelta(seconds=10))
        store.set("test_ip_port", "value")
        data = store.get("test_ip_port")
        assert data.decode() == "value"
    barrier()
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+6, is_master=False, timeout=timedelta(seconds=10))
        data = store.get("test_ip_port")
        assert data.decode() == "value"
        ret = store.delete_key("test_ip_port")
        assert ret is True
        ret = store.delete_key("test_ip_port")
        assert ret is False
    barrier()


@log_function_entry_exit
def test_add():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+7, is_master=True, timeout=timedelta(seconds=10))
        store.set("test_ip_port", "3")
        data = store.add("test_ip_port", -3)
        assert data == 0
    barrier()
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+7, is_master=False, timeout=timedelta(seconds=10))
        data = store.get("test_ip_port")
        assert data.decode() == "0"
        data = store.add("test_ip_port", -3)
        assert data == -3
        data = store.add("test_ip_port", -3)
        assert data == -6
    barrier()


@log_function_entry_exit
def test_ip_port_get1():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        import time
        time.sleep(2)
        store = TCPStore("127.0.0.1", start_port+8, is_master=True, timeout=timedelta(seconds=10))
        time.sleep(3)
        store.set("test_ip_port", "value")
        data = store.get("test_ip_port")
        assert data.decode() == "value"
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+8, is_master=False, timeout=timedelta(seconds=10))
        data = store.get("test_ip_port")
        assert data.decode() == "value"
        ret = store.delete_key("test_ip_port")
        assert ret is True
        ret = store.delete_key("test_ip_port")
        assert ret is False
    barrier()


@log_function_entry_exit
def test_ip_port_get2():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+9, is_master=True, timeout=timedelta(seconds=300))
        value = 'A' * 1024 * 1024
        store.set("test_ip_port_get2", value)
    if this_rank != 0:
        store = TCPStore("127.0.0.1", start_port+9, is_master=False, timeout=timedelta(seconds=300))
        data = store.get("test_ip_port_get2")
        assert len(data) == 1024 * 1024
        assert data[0] == 65
    barrier()


@log_function_entry_exit
def test_ip_port_wait_for_workers1():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    barrier()
    import time
    start = time.time()
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+10, 2, True, timedelta(seconds=5), False)
        store.set("test_ip_port_wait_for_workers1", "value")
    barrier()
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+10, 2, False, timedelta(seconds=5), False)
        data = store.get("test_ip_port_wait_for_workers1")
        assert data.decode() == "value"
    end = time.time()
    t = end - start
    assert t < 5 - 0.1
    barrier()


@log_function_entry_exit
def test_ip_port_wait_for_workers2():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    barrier()
    import time
    start = time.time()
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+11, 2, True, timedelta(seconds=5), True)
        store.set("test_ip_port_wait_for_workers2", "value")
    barrier()
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+11, 2, False, timedelta(seconds=5), True)
        data = store.get("test_ip_port_wait_for_workers2")
        assert data.decode() == "value"
    end = time.time()
    t = end - start
    assert t > 5 - 0.1
    barrier()


@log_function_entry_exit
def test_ip_port_get3():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    barrier()
    import time
    start = time.time()
    if this_rank == 0:
        time.sleep(5)
        store = TCPStore("127.0.0.1", start_port+12, is_master=True, timeout=timedelta(seconds=30))
        store.set("test_ip_port_get3", "value")
    if this_rank != 0:
        store = TCPStore("127.0.0.1", start_port+12, is_master=False, timeout=timedelta(seconds=30))
        data = store.get("test_ip_port_get3")
        assert data.decode() == "value"
    end = time.time()
    t = end - start
    assert t > 5 - 0.1
    barrier()


@log_function_entry_exit
def test_ip_port_wait_for_workers3():
    """
    Feature: test distributed op
    Description: test tcp store in python native
    Expectation: success
    """
    barrier()
    import time
    start = time.time()
    if this_rank == 0:
        time.sleep(3)
        store = TCPStore("127.0.0.1", start_port+13, 2, True, timedelta(seconds=5), True)
        store.set("test_ip_port_wait_for_workers3", "value")
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+13, 2, False, timedelta(seconds=5), True)
        data = store.get("test_ip_port_wait_for_workers3")
        assert data.decode() == "value"
    end = time.time()
    t = end - start
    if this_rank == 0 or this_rank == 1:
        assert t > 3 - 0.1
    barrier()


@log_function_entry_exit
def test_tcp_complete001():
    """
    Feature: test distributed op
    Description: port is diff
    Expectation: success
    """
    store = None
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+14, None, True, timedelta(seconds=5), True)
        store.set("test_tcp_complete001", "value")
    if this_rank != 0:
        with pytest.raises(RuntimeError):
            store = TCPStore("127.0.0.1", start_port+15, None, False, timedelta(seconds=5), True)
        store = TCPStore("127.0.0.1", start_port+14, None, False, timedelta(seconds=5), True)
        data = store.get("test_tcp_complete001")
        assert data.decode() == "value"
    barrier()


@log_function_entry_exit
def test_tcp_complete002():
    """
    Feature: test distributed op
    Description: port is diff
    Expectation: success
    """
    store = None
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+16, None, True, timedelta(seconds=5), True)
        store.set("test_tcp_complete001", "value")
        with pytest.raises(RuntimeError):
            TCPStore("127.0.0.1", start_port+16, None, True, timedelta(seconds=5), True)
    barrier()


@log_function_entry_exit
def test_tcp_complete003():
    """
    Feature: test distributed op
    Description: world_size is err
    Expectation: success
    """
    with pytest.raises(ValueError):
        TCPStore("127.0.0.1", start_port+17, 0, True, timedelta(seconds=5), True)
    with pytest.raises(ValueError):
        TCPStore("127.0.0.1", start_port+17, -1, True, timedelta(seconds=5), True)
    with pytest.raises(ValueError):
        TCPStore("127.0.0.1", -1, 1, True, timedelta(seconds=5), True)
    with pytest.raises(ValueError):
        TCPStore("127.0.0.1", 65536, 1, True, timedelta(seconds=5), True)
    barrier()


@log_function_entry_exit
def test_tcp_complete004():
    """
    Feature: test distributed op
    Description: hostname is None
    Expectation: success
    """
    with pytest.raises(TypeError):
        TCPStore(None, start_port+17, 1, True, timedelta(seconds=5), True)
    barrier()


@log_function_entry_exit
def test_tcp_complete005():
    """
    Feature: test distributed op
    Description: add para is err
    Expectation: success
    """
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+18, is_master=True, timeout=timedelta(seconds=5))
        with pytest.raises(TypeError):
            store.add(123, -3)
        with pytest.raises(TypeError):
            store.add("test_tcp_complete005", 1.5)
    barrier()
    if this_rank == 1:
        store = TCPStore("127.0.0.1", start_port+18, is_master=False, timeout=timedelta(seconds=5))
        with pytest.raises(TypeError):
            store.add(123, -3)
        with pytest.raises(TypeError):
            store.add("test_tcp_complete005", 1.5)
    barrier()


@log_function_entry_exit
def test_tcp_complete006():
    """
    Feature: test distributed op
    Description: worker timeout
    Expectation: success
    """
    with pytest.raises(RuntimeError):
        TCPStore("127.0.0.1", start_port+19, is_master=False, timeout=timedelta(seconds=5))
    barrier()


@log_function_entry_exit
def test_ip_port_get4():
    """
    Feature: test distributed op
    Description: test add exception
    Expectation: success
    """
    if this_rank == 0:
        store = TCPStore("127.0.0.1", start_port+20, is_master=True, timeout=timedelta(seconds=30))
        store.set("test_ip_port_get4", "value")
        with pytest.raises(RuntimeError):
            store.add('test_ip_port_get4', 2)
    barrier()
    if this_rank != 0:
        store = TCPStore("127.0.0.1", start_port+20, is_master=False, timeout=timedelta(seconds=30))
        data = store.get("test_ip_port_get4")
        assert data.decode() == "value"
    barrier()
