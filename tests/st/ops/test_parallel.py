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

import os
import pathlib
import sys
import threading
import socket

from tests import mark_utils

if sys.version_info >= (3, 9):
    from typing import Callable
    dict_annotation, list_annotation, tuple_annotation = dict, list, tuple
else:
    from typing import Callable, Dict, List, Tuple
    dict_annotation, list_annotation, tuple_annotation = Dict, List, Tuple


_WORKER_NUM = 2
_ALL_GATHER_MATMUL_TEST_SCRIPT = 'parallel/all_gather_matmul.py'
_MATMUL_REDUCE_SCATTER_TEST_SCRIPT = 'parallel/matmul_reduce_scatter.py'
_TEST_BINARY_CASE = 'test_binary_case'
_TEST_DYNAMIC_SHAPE = 'test_dynamic_shape'
_TEST_NORMAL = 'test_precision_with_ms_small_ops'
_PARALLEL_STRATEGY = [
    ('_init0', '0,1', '61000'),
    ('_init1', '2,3', '62000'),
    ('_init2', '4,5', '63000'),
]


class TestThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception = None

    def run(self):
        try:
            super().run()
        except RuntimeError as e:
            self.exception = e


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def run_test(
        test_script: str,
        test_name: str,
        devices: str,
        filter_: str,
        hccl_port: str,
        env_vars: dict_annotation[str, str] = None,
    ) -> None:
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = devices
    if env_vars:
        for key, value in env_vars.items():
            os.environ[key] = value

    port = get_free_port()
    path = pathlib.Path(test_script)
    log_dir = path.parent / path.stem / (test_name + filter_)
    status = os.system(rf'''
        export ASCEND_RT_VISIBLE_DEVICES={devices} \
        && export HCCL_IF_BASE_PORT={hccl_port} \
        && msrun \
            --worker_num {_WORKER_NUM} \
            --local_worker_num {_WORKER_NUM} \
            --join True \
            --master_port {port} \
            --log_dir {log_dir.as_posix()} \
            pytest -vra --disable-warnings {test_script}::{test_name}
    ''')
    if status != 0:
        raise RuntimeError(f'Test failed with status {status}, please check {log_dir.as_posix()} for more details.')


def run_all_gather_matmul_test(test_script: str, test_name: str, devices: str, filter_: str, hccl_port: str) -> None:
    run_test(test_script, test_name, devices, filter_, hccl_port)


def run_matmul_reduce_scatter_test(
        test_script: str,
        test_name: str,
        devices: str,
        filter_: str,
        hccl_port: str,
    ) -> None:
    run_test(test_script, test_name, devices, filter_, hccl_port, {'HCCL_DETERMINISTIC': 'true'})


def run_parallel_tests(
        test_script: str,
        test_name: str,
        parallel_strategy: list_annotation[tuple_annotation[str, str, str]],
        test_func: Callable[[str, str, str, str], None],
    ) -> None:
    threads = []
    for filter_, devices, hccl_port in parallel_strategy:
        thread = TestThread(target=test_func, args=(test_script, test_name, devices, filter_, hccl_port))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
        if thread.exception is not None:
            raise thread.exception


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='allcards')
def test_all_gather_matmul_binary_case() -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
        torch_npu.npu_all_gather_base_mm forword calculation.
    """
    run_parallel_tests(
        _ALL_GATHER_MATMUL_TEST_SCRIPT,
        _TEST_BINARY_CASE,
        _PARALLEL_STRATEGY,
        run_all_gather_matmul_test,
    )


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='allcards')
def test_all_gather_matmul_dynamic_shape() -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the dynamic shape function of forward calculation.
    Expectation: The result of forward calculation with inputs in dynamic shapes is equal to the result of
        forword calculation with inputs in static shapes.
    """
    run_parallel_tests(
        _ALL_GATHER_MATMUL_TEST_SCRIPT,
        _TEST_DYNAMIC_SHAPE,
        _PARALLEL_STRATEGY,
        run_all_gather_matmul_test,
    )


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='allcards')
def test_all_gather_matmul_precision_with_ms_small_ops() -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
        mindspore.ops.AllGahter and mindspore.ops.MatMul forword calculation.
    """
    run_parallel_tests(
        _ALL_GATHER_MATMUL_TEST_SCRIPT,
        _TEST_NORMAL,
        _PARALLEL_STRATEGY,
        run_all_gather_matmul_test,
    )


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='allcards')
def test_matmul_reduce_scatter_binary_case() -> None:
    """
    Feature: mindspore.ops.matmul_reduce_scatter
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.matmul_reduce_scatter forward calculation is equal to the result of
        torch_npu.npu_mm_reduce_scatter_base forword calculation.
    """
    run_parallel_tests(
        _MATMUL_REDUCE_SCATTER_TEST_SCRIPT,
        _TEST_BINARY_CASE,
        _PARALLEL_STRATEGY,
        run_matmul_reduce_scatter_test,
    )


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='allcards')
def test_matmul_reduce_scatter_dynamic_shape() -> None:
    """
    Feature: mindspore.ops.matmul_reduce_scatter
    Description: Test the dynamic shape function of forward calculation.
    Expectation: The result of forward calculation with inputs in dynamic shapes is equal to the result of
        forword calculation with inputs in static shapes.
    """
    run_parallel_tests(
        _MATMUL_REDUCE_SCATTER_TEST_SCRIPT,
        _TEST_DYNAMIC_SHAPE,
        _PARALLEL_STRATEGY,
        run_matmul_reduce_scatter_test,
    )


@mark_utils.arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='allcards')
def test_matmul_reduce_scatter_precision_with_ms_small_ops() -> None:
    """
    Feature: mindspore.ops.matmul_reduce_scatter
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.matmul_reduce_scatter forward calculation is equal to the result of
        mindspore.ops.MatMul and mindspore.ops.ReduceScatter forword calculation.
    """
    run_parallel_tests(
        _MATMUL_REDUCE_SCATTER_TEST_SCRIPT,
        _TEST_NORMAL,
        _PARALLEL_STRATEGY,
        run_matmul_reduce_scatter_test,
    )
