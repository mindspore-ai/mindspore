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
"""
The tests of mindspore, used to test aclgraph.
"""
import os
import pytest
import subprocess
from tests.st.runtime.kernel_capture.run_capture_graph import Net1, SeqNet, expected_output, SimpleWrapperNet
from tests.mark_utils import arg_mark
import numpy as np
import mindspore.ops as P
import mindspore.runtime as rt
from mindspore import Tensor, context, mutable
from mindspore import dtype as mstype

context.set_context(
    mode=context.GRAPH_MODE,
    jit_config={
        "jit_level": "O0",
        "infer_boost": "on"
    },
    max_call_depth=600000
)

g_block_num = 20
steps = 20
input_len = 10

@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential', mem_peak=6.01
)
def test_dynamic_shape_for_capture_graph():
    """
    Feature: graph mode support capture graph
    Description: Test dynamic shape scene and dyn value for capture graph
    Expectation: No exception and result is correct
    """
    rt.set_kernel_launch_capture(True)
    new_input1 = Tensor(np.ones((2, 5)).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    base_shape = (2, 3)

    net = SeqNet()
    net.set_inputs(dyn_input_data)
    net.phase = "increment"

    for i in range(1, 20):
        if i == 5:
            output = net(new_input1)
            output_np = output.asnumpy()
        else:
            input_data1 = Tensor(np.full(base_shape, i).astype(np.float32))
            output = net(input_data1)
            output_np = output.asnumpy()
            expected = expected_output(i)
            assert np.allclose(output_np, expected), \
                f"Output {output_np} does not match expected {expected} at step {i}"

@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential', mem_peak=6.01
)
def test_dynamic_shape_with_view_ops_for_capture_graph():
    """
    Feature: graph mode support capture graph
    Description: Test view op in aclgraph
    Expectation: No exception and results are correct at each step
    """
    rt.set_kernel_launch_capture(True)

    ori_shape = (2, 3)
    fixed_shape = (3, 2)
    ori_input_data = Tensor(np.ones(ori_shape).astype(np.float32))

    dyn_input = Tensor(shape=[3, None], dtype=mstype.float32)

    net = SimpleWrapperNet()
    net.set_inputs(dyn_input)
    net.phase = "increment"

    param = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    expected = np.ones(fixed_shape) + param

    results = []
    input_data = P.Transpose()(ori_input_data, (1, 0))
    for step in range(1, 4):
        output = net(input_data)
        output_np = output.asnumpy()

        assert np.allclose(output_np, expected), \
            f"At step {step}, output {output_np} does not match expected {expected}"

        print(f"Step {step} passed, output shape: {output_np.shape}")
        results.append(output_np)

    return results

@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential', mem_peak=6.01
)
def test_kv_cache_for_capture_graph():
    """
    Feature: graph mode support capture graph
    Description: Test kv_cache scene
    Expectation: No exception and result is correct
    """
    rt.set_kernel_launch_capture(True, ['reshape'])
    input_data1 = Tensor(np.zeros((2, 2)).astype(np.float32))
    input_data2 = Tensor(np.zeros((2, 4)).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    k_cache_list1 = []
    v_cache_list1 = []
    k_cache_list2 = []
    v_cache_list2 = []
    dyn_k_cache_list = []
    dyn_v_cache_list = []

    for _ in range(input_len):
        dyn_k_cache_list.append(dyn_input_data)
        dyn_v_cache_list.append(dyn_input_data)

    for _ in range(input_len):
        new_input_data = P.Add()(input_data1, 1)
        new_input_data_view = new_input_data.view(new_input_data.shape)
        k_cache_list1.append(new_input_data_view)
        v_cache_list1.append(new_input_data_view)

    net = Net1()
    net.set_inputs(dyn_input_data, mutable(dyn_k_cache_list), mutable(dyn_v_cache_list))
    net.phase = "increment"

    output = net(input_data1, mutable(k_cache_list1), mutable(v_cache_list1))
    output = net(input_data1, mutable(k_cache_list1), mutable(v_cache_list1))

    k_cache_list1 = []
    v_cache_list1 = []

    for _ in range(input_len):
        new_input_data = P.Add()(input_data2, 1)
        k_cache_list2.append(new_input_data)
        v_cache_list2.append(new_input_data)

    for _ in range(steps):
        output = net(input_data2, mutable(k_cache_list2), mutable(v_cache_list2))
        output.asnumpy()

    expected = np.array([[11.036, 11.036, 11.036, 11.036], [11.036, 11.036, 11.036, 11.036]], dtype=np.float32)

    assert np.allclose(output, expected, rtol=0, atol=0.001), f"Result wrong, real: {output}, expected: {expected}"


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=0.01
)
def test_multi_graph_cache_for_capture_graph():
    """
    Feature: capture graph support multi graph cache
    Description: Test capture count and replay count are both correct
    Expectation: No exception and result is correct
    """
    expected_capture_count = 3
    expected_replay_count = 11

    command = 'export GLOG_v=1 && python run_capture_graph.py > capture_graph.log 2>&1'
    os.system(command)

    log_file = "capture_graph.log"

    try:
        capture_count = int(subprocess.check_output(
            f"grep 'Begin launch all kernels with capture graph' {log_file} | wc -l",
            shell=True
        ).decode().strip())

        replay_count = int(subprocess.check_output(
            f"grep 'Begin launch all kernels with replay graph' {log_file} | wc -l",
            shell=True
        ).decode().strip())

        assert capture_count == expected_capture_count, \
            f"Expected {expected_capture_count} capture graph launched, got {capture_count}"
        assert replay_count == expected_replay_count, \
            f"Expected {expected_replay_count} capture graph launched, got {replay_count}"

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to analyse logs: {str(e)}")
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=0.01
)
def test_multi_graph_cache_with_num_limit_for_capture_graph():
    """
    Feature: capture graph add env to control max capture limit
    Description: Test capture count and replay count are both correct
    Expectation: No exception and result is correct
    """
    expected_capture_count = 2
    expected_replay_count = 10

    os.environ["MS_DEV_RUNTIME_CONF"] = "graph_capture_max_number:2"

    command = 'export GLOG_v=1 && python run_capture_graph.py > capture_graph_num_limit.log 2>&1'
    os.system(command)

    log_file = "capture_graph_num_limit.log"

    try:
        capture_count = int(subprocess.check_output(
            f"grep 'Begin launch all kernels with capture graph' {log_file} | wc -l",
            shell=True
        ).decode().strip())

        replay_count = int(subprocess.check_output(
            f"grep 'Begin launch all kernels with replay graph' {log_file} | wc -l",
            shell=True
        ).decode().strip())

        assert capture_count == expected_capture_count, \
            f"Expected {expected_capture_count} capture graph launched, got {capture_count}"
        assert replay_count == expected_replay_count, \
            f"Expected {expected_replay_count} capture graph launched, got {replay_count}"

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to analyse logs: {str(e)}")
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=0.01
)
def test_weight_change():
    """
    Feature: test weight change
    Description: Weight input change will trigger exception when enable capture graph
    Expectation: give relative exception
    """
    command = 'export GLOG_v=1 && python run_capture_graph_with_exception.py > capture_graph_weight_change.log 2>&1'
    os.system(command)

    log_file = "capture_graph_weight_change.log"

    try:
        address_changed = int(subprocess.check_output(
            f"grep 'device address has changed' {log_file} | wc -l",
            shell=True
        ).decode().strip())

        assert address_changed > 0, "Expected 'device address has changed' in logs but not found"

    except subprocess.CalledProcessError as e:
        print(f"Failed to analyse logs: {str(e)}")
        raise
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential', mem_peak=4.01
)
def test_aclgraph_dyn_input_off():
    """
    Feature: Disable aclgraph in dynamic KV cache scenario
    Description: Test dynamic shape incremental network with varying inputs per round, 
    containing host-bound and non-capturable operators, single card with aclgraph disabled
    Expectation: No exception and result is correct
    """
    rt.set_kernel_launch_capture(False)
    base_shape = (2, 3)
    input_data1 = Tensor(np.ones((2, 5)), mstype.float32)
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    net = SeqNet()
    net.set_inputs(dyn_input_data)
    net.phase = "increment"
    for i in range(1, 500):
        if i == 5:
            output = net(input_data1)
            output_np = output.asnumpy()
        else:
            input_data2 = Tensor(np.full(base_shape, i).astype(np.float32))
            output = net(input_data2)
            output_np = output.asnumpy()
            expected = expected_output(i)
            assert np.allclose(output_np, expected), \
                f"Output {output_np} does not match expected {expected} at step {i}"


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=0.01
)
def test_aclgraph_api():
    """
    Feature: Non-boolean parameter validation
    Description: Test passing non-boolean values to graph capture interface
    Expectation: Raise appropriate exception
    """
    with pytest.raises(TypeError, match="The parameter 'enable_capture_graph' must be <class 'bool'>"):
        rt.set_kernel_launch_capture("True")


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
    mem_peak=0.01
)
def test_aclgraph_group():
    """
    Feature: Conflict between graph capture and parallel launch
    Description: Test aclgraph conflicts with parallel launch
    Expectation: Raise appropriate exception
    """
    with pytest.raises(RuntimeError):
        rt.set_kernel_launch_capture(True)
        rt.set_kernel_launch_group(thread_num=2, kernel_group_num=8)


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential', mem_peak=4.01
)
def test_set_enable_capture_graph_in_wrong_time():
    """
    Feature: test aclgraph
    Description: Test set aclgraph not the first time
    Expectation: No exception and result is correct
    """
    new_input1 = Tensor(np.ones((2, 5)).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    base_shape = (2, 3)

    net = SeqNet()
    net.set_inputs(dyn_input_data)
    net.phase = "increment"

    input_data1 = Tensor(np.full(base_shape, 1).astype(np.float32))
    net(input_data1)
    rt.set_kernel_launch_capture(True)

    with pytest.raises(RuntimeError) as e:
        net(new_input1)
        net(new_input1)
    assert "set up the ACL graph before the first step" in str(e.value)
