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
"""Test cases for compilation cache functionality.
   This module contains test cases to verify the compilation cache feature in MindSpore.
   It tests various scenarios including normal networks in train or inference scenarios, jit func,
   control flow, kernel packet, lazy inline, and different compilation configurations.
"""
import numpy as np
import glob
import os
import re
import shutil
import subprocess
import hashlib

from mindspore import context
from tests.mark_utils import arg_mark

context.set_context(device_target="Ascend")

match_output = re.compile(r'AAA(.*?)BBB', re.S)
match_num = re.compile(r'\d+\.?\d*', re.S)


def clear_comple_cache_files_before_run_case(cache_path, log_file_name_first, log_file_name_second):
    """Clear compilation cache directory and log files before test execution.
       Removes existing cache folder and log files to ensure clean test environment.
       Verifies that all specified paths are successfully deleted.

    Args:
        cache_path (str): Path to compilation cache directory
        log_file_name_first (str): Path to first log file
        log_file_name_second (str): Path to second log file
    """
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    if os.path.exists(log_file_name_first):
        os.remove(log_file_name_first)
    if os.path.exists(log_file_name_second):
        os.remove(log_file_name_second)

    assert not os.path.exists(cache_path)
    assert not os.path.exists(log_file_name_first)
    assert not os.path.exists(log_file_name_second)


def check_backend_compile_cache_files(cache_path):
    assert os.path.exists(cache_path)
    # Funcgraph
    assert os.path.exists(cache_path + "/rank_0/graph_cache/compile_dependency.hash")
    assert os.path.exists(cache_path + "/rank_0/graph_cache/compile_cache_0.mindir")
    # Kernelgraph
    matching_files = glob.glob(cache_path + "/rank_0/graph_cache/backend_compile_cache_0*.mindir")
    assert os.path.exists(cache_path + "/rank_0/graph_cache/backend_compile_cache_0.json")
    assert len(matching_files) > 0


def get_hash_file_md5(cache_path):
    hash_file_path = cache_path + "/rank_0/graph_cache/compile_dependency.hash"
    assert os.path.exists(hash_file_path)
    hash_alg = hashlib.md5()
    with open(hash_file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_alg.update(chunk)
    computed_hash = hash_alg.hexdigest()
    return computed_hash


def check_hash_file(cache_path, first_run_hash):
    assert os.path.exists(cache_path)
    hash_file_path = cache_path + "/rank_0/graph_cache/compile_dependency.hash"
    assert os.path.exists(hash_file_path)
    hash_alg = hashlib.md5()
    with open(hash_file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_alg.update(chunk)
    computed_hash = hash_alg.hexdigest()
    return computed_hash == first_run_hash


def get_compile_times_log(compile_times):
    # The default compile times is one.
    backend_compile_time_cost_log = [1, 1]
    # Compile graph twice
    if compile_times == "twice":
        backend_compile_time_cost_log = [2, 2]
    # Compile graph four times
    elif compile_times == "four_times":
        backend_compile_time_cost_log = [4, 4]
    elif compile_times == "special":
        backend_compile_time_cost_log = [4, 3]
    elif compile_times == "temp":
        backend_compile_time_cost_log = [3, 2]
    return backend_compile_time_cost_log


def run_case_with_kbk_mode(file_name):
    """
    Only support kbk
    """
    assert os.path.exists(file_name)
    temp_file = file_name + ".tp.py"
    with open(file_name, "r", encoding="utf-8") as file:
        ctx = file.read()
    ctx = ctx.replace("O2", "O0")
    with open(temp_file, "w", encoding="utf-8") as file:
        file.write(ctx)
    assert os.path.exists(temp_file)
    return temp_file


def check_lazy_inline_log(file_name, log_file):
    if "with_lazy_inline" in file_name:
        with open(log_file, "r", encoding="utf-8") as file:
            content = file.read()
        assert "can be lazyinlined" in content
    elif "without_lazy_inline" in file_name:
        with open(log_file, "r", encoding="utf-8") as file:
            content = file.read()
        assert "can be lazyinlined" not in content


def check_kernel_packet_log(file_name, log_file, is_first_run):
    """Verify kernel packet related logs based on test case type and run sequence.
       Checks for presence or absence of kernel packet compilation and execution logs
       depending on whether the test case uses kernel packet feature and if it's the first run.

    Args:
        file_name (str): Test script filename to determine test type
        log_file (str): Path to the log file to check
        is_first_run (bool): True if checking first run logs, False for second run
    """
    if "with_kernel_packet" in file_name and is_first_run:
        with open(log_file, "r", encoding="utf-8") as file:
            content = file.read()
        # First run check kernel packet is enabled in compile cache.
        assert "symbol_engine_extender is enabled" in content
    elif "without_kernel_packet" in file_name:
        with open(log_file, "r", encoding="utf-8") as file:
            content = file.read()
        assert "symbol_engine_extender is enabled" not in content


def check_switch_inline_log(file_name, log_file):
    """Verify switch inline related logs based on test case type and run sequence.
    Check if switch inline is enabled or disabled depending on switch inline supported case.

    Args:
        file_name (str): Test script filename to determine test type
        log_file (str): Path to the log file to check
    """
    switch_inline_network = ["run_net_if_any", "run_net_for_after_for_in_if", "run_partial_without_input"]
    if any(key_file_name in file_name for key_file_name in switch_inline_network):
        with open(log_file, "r", encoding="utf-8") as file:
            content = file.read()
        assert "IsValidFuncGraph] Enable Switch Inline" in content


def run_twice_with_same_network(file_name, cache_path, log_file_name_first, log_file_name_second, compile_times):
    """Run the same network twice to test compile cache. First run generates compicompileation cache,
       second run uses cache and verifies result consistency. Validates cache generation, hash consistency,
       and output matching.

    Args:
        file_name (str): Path to the test script file
        cache_path (str): Directory path for compilation cache storage
        log_file_name_first (str): Log file path for the first execution
        log_file_name_second (str): Log file path for the second execution
        compile_times (int): Expected number of compilation operations
    """
    current_match_output = match_output
    if "compiler" not in file_name:
        current_match_output = re.compile(r'RUNTIME_COMPILE(.*?)RUNTIME_CACHE', re.S)

    backend_compile_time_cost_log = get_compile_times_log(compile_times)
    clear_comple_cache_files_before_run_case(cache_path, log_file_name_first, log_file_name_second)
    temp_file = run_case_with_kbk_mode(file_name)

    os.environ['MS_DEV_RUNTIME_CONF'] = "memory_statistics:True,compile_statistics:True,backend_compile_cache:True"
    # First run without compile cache
    cmd_first = "export GLOG_v=1; export MS_COMPILER_CACHE_ENABLE=1; " \
                + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, temp_file,
                                                                                 log_file_name_first)

    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r", encoding="utf-8") as f_first:
        data_first = f_first.read()

    # First run check compile cache end and save compile cache
    assert "Status record: Start cache backend kernel graph." in data_first
    assert "Status record: End cache backend kernel graph and control node info." in data_first
    assert data_first.count("] [PROF]compile_backend_graph cost") == backend_compile_time_cost_log[0]
    assert "Status record: end compile function graph:" in data_first
    # Check lazy inline log
    check_lazy_inline_log(temp_file, log_file_name_first)
    # Check kernel packet log
    check_kernel_packet_log(temp_file, log_file_name_first, True)
    # Check switch inline log
    check_switch_inline_log(temp_file, log_file_name_first)

    # Take out the result of the first run
    match_output_first = re.findall(current_match_output, data_first)
    assert len(match_output_first) == 2
    nums_first = re.findall(match_num, match_output_first[0])
    array_first = np.array([float(x) for x in nums_first])
    shape_first = re.findall(match_num, match_output_first[1])
    array_shape_first = np.array([int(x) for x in shape_first])

    # Check .mindir .json before second run with compile cache
    check_backend_compile_cache_files(cache_path)
    # Check hash is same before second run with compile cache
    first_run_hash = get_hash_file_md5(cache_path)
    assert check_hash_file(cache_path, first_run_hash)

    # Second run with compile cache
    cmd_second = "export GLOG_v=1; export MS_COMPILER_CACHE_ENABLE=1; " \
                 + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, temp_file,
                                                                                  log_file_name_second)

    subprocess.check_output(cmd_second, shell=True)
    assert os.path.exists(log_file_name_second)
    with open(log_file_name_second, "r", encoding="utf-8") as f_second:
        data_second = f_second.read()

    # Second run can compile cache
    assert "Enable backend compile cache." in data_second
    assert "Status record: Start load backend kernel graph." in data_second
    assert "Status record: start use cache to compile graph kbk." in data_second
    assert data_second.count("] [PROF]Load_backend_compile_cache") == backend_compile_time_cost_log[1]
    # Check lazy inline log
    check_lazy_inline_log(temp_file, log_file_name_second)
    # Check kernel packet log
    check_kernel_packet_log(temp_file, log_file_name_second, False)
    # Check switch inline log
    check_switch_inline_log(temp_file, log_file_name_first)

    # Take out the result of the second run
    match_output_second = re.findall(current_match_output, data_second)
    assert len(match_output_second) == 2
    nums_second = re.findall(match_num, match_output_second[0])
    array_second = np.array([float(x) for x in nums_second])
    shape_second = re.findall(match_num, match_output_second[1])
    array_shape_second = np.array([int(x) for x in shape_second])

    assert np.allclose(array_first, array_second, 0.0001, 0.0001)
    assert (array_shape_first == array_shape_second).all()

    # Clean files
    os.remove(log_file_name_first)
    os.remove(log_file_name_second)
    os.remove(temp_file)
    shutil.rmtree(cache_path)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_simple_net():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function can run successfully.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_simple_net.py"
    run_twice_with_same_network(pypath, "./simple_net", "simple_net_first.txt", "simple_net_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_lenet():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function can run successfully.
    Expectation: Run success.
    """
    st_path = os.path.realpath(os.path.join(os.getcwd(), "../../.."))
    pypath = st_path + "/compiler/compile_cache/run_lenet.py"
    run_twice_with_same_network(pypath, "./lenet", "lenet_first.txt", "lenet_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_simple_jit_func():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache function can run successfully in the compilation of ms_function.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_simple_jit_func.py"
    run_twice_with_same_network(pypath, "./simple_jit_func", "simple_jit_func_first.txt",
                                "simple_jit_func_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_resnet_infer():
    """
    Feature: Compile cache in inference scenarios.
    Description: Test compile cache in inference scenarios.
    Expectation: Run success.
    """
    st_path = os.path.realpath(os.path.join(os.getcwd(), "../../.."))
    pypath = st_path + "/compiler/compile_cache/run_resnet_infer.py"
    run_twice_with_same_network(pypath, "./resnet_infer", "resnet_infer_first.txt",
                                "resnet_infer_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_with_range_kernel_packet():
    """
    Feature: Compile cache with range  cache in inference scenarios.
    Description: Test compile cache with range kernel packet scenarios.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_with_kernel_packet_range.py"
    run_twice_with_same_network(pypath, "./kernel_packet_range", "kernel_packet_range_first.txt",
                                "kernel_packet_range_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_simple_net_change_dir():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function can run successfully when changing
                 the current work directory.
    Expectation: Run success.
    """
    cwd = os.getcwd()
    new_path = cwd + '/tmp'
    os.mkdir(new_path)
    os.chdir(new_path)
    run_twice_with_same_network("../run_simple_net.py", "../simple_net_change_dir",
                                "../simple_net_change_dir_first.txt", "../simple_net_change_dir_second.txt", "once")
    shutil.rmtree(new_path, ignore_errors=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_partial_without_inputs():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache function can run successfully for the graph with a partial node
                 without inputs.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_partial_without_input.py"
    run_twice_with_same_network(pypath, "./partial_without_input", "partial_without_input_first.txt",
                                "partial_without_input_second.txt", "twice")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_if_any():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache function can run successfully when in control flow scenarios.
                 true-branch func graph refer a CNode in construct as free variable.
                 That CNode and the inputs will be specialized before ProcessCNode
                 of true-branch func graph, so it's no need to specialize the inputs
                 of that CNode again if it's a specialized func graph.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_if_any.py"
    run_twice_with_same_network(pypath, "./if_any", "if_any_first.txt", "if_any_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_for_n_while():
    """
    Feature: Compile cache.
    Description: Test compile cache in control flow with grad jit scenarios.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_for_n_while.py"
    run_twice_with_same_network(pypath, "./for_n_while", "for_n_while_first.txt", "for_n_while_second.txt", "twice")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_for_for_while():
    """
    Feature: Compile cache.
    Description: Test compile cache in control flow with grad jit scenarios.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_for_for_while.py"
    run_twice_with_same_network(pypath, "./for_for_while", "for_for_while_first.txt",
                                "for_for_while_second.txt", "twice")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_for_after_for_in_if():
    """
    Feature: Compile cache.
    Description: Test compile cache in control flow with grad jit scenarios.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_for_after_for_in_if.py"
    run_twice_with_same_network(pypath, "./for_after_for_in_if", "for_after_for_in_if_first.txt",
                                "for_after_for_in_if_second.txt", "four_times")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_with_lazy_inline():
    """
    Feature: Compile cache with lazy inline.
    Description: Test compile cache when all inline in single graph.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_with_lazy_inline.py"
    run_twice_with_same_network(pypath, "./net_lazy_inline", "net_lazy_inline_first.txt",
                                "net_lazy_inline_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_without_lazy_inline():
    """
    Feature: Compile cache without lazy inline.
    Description: Test compile cache when lazy inline is disabled.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_with_lazy_inline.py"
    assert os.path.exists(pypath)
    new_pypath = fpath + "/compile_cache/run_net_without_lazy_inline.py"
    shutil.copy2(pypath, new_pypath)

    # Disable kernel packet by environ
    with open(new_pypath, 'r', encoding="utf-8") as file:
        content = file.read()
    assert "@lazy_inline" in content
    new_content = content.replace("@lazy_inline", "")
    with open(new_pypath, 'w', encoding="utf-8") as file:
        file.write(new_content)
    # Check whether kernel packet is disabled
    with open(new_pypath, 'r', encoding="utf-8") as file:
        final_content = file.read()
    assert "@lazy_inline" not in final_content
    run_twice_with_same_network(new_pypath, "./net_without_lazy_inline", "net_without_lazy_inline_first.txt",
                                "net_without_lazy_inline_second.txt", "once")
    if os.path.exists(new_pypath):
        os.remove(new_pypath)
    assert not os.path.exists(new_pypath)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_with_reducesum_kernel_packet():
    """
    Feature: Compile cache with kernel packet.
    Description: Test compile cache when enable kernelpacket.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_with_kernel_packet_reducesum.py"
    run_twice_with_same_network(pypath, "./kernel_packet_reducesum", "kernel_packet_reducesum_first.txt",
                                "kernel_packet_reducesum_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_without_kernel_packet():
    """
    Feature: Compile cache without kernel packet.
    Description: Test compile cache when disable kernel packet.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_with_kernel_packet_reducesum.py"
    assert os.path.exists(pypath)
    new_pypath = fpath + "/compile_cache/run_without_kernel_packet_reducesum.py"
    shutil.copy2(pypath, new_pypath)

    # Disable kernel packet by environ
    with open(new_pypath, 'r', encoding="utf-8") as file:
        content = file.read()
    assert 'jit_level="O1"' in content
    new_content = content.replace('jit_level="O1"', 'jit_level="O0"')
    with open(new_pypath, 'w', encoding="utf-8") as file:
        file.write(new_content)
    # Check whether kernel packet is disabled
    with open(new_pypath, 'r', encoding="utf-8") as file:
        final_content = file.read()
    assert 'jit_level="O0"' in final_content
    assert 'jit_level="O1"' not in final_content
    run_twice_with_same_network(new_pypath, "./net_without_kernel_packet", "net_without_kernel_packet_first.txt",
                                "net_without_kernel_packet_second.txt", "once")
    if os.path.exists(new_pypath):
        os.remove(new_pypath)

    assert not os.path.exists(new_pypath)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_for_in_if():
    """
    Feature: Compile cache.
    Description: Test compile cache when func graph name changed by backend_pass.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_for_in_if.py"
    run_twice_with_same_network(pypath, "./for_in_if", "for_in_if_first.txt", "for_in_if_second.txt", "twice")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_simple_jit_net():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache function can run successfully in the compilation of ms_function.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_simple_jit_net.py"
    run_twice_with_same_network(pypath, "./simple_jit", "simple_jit_first.txt", "simple_jit_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_bprop_grad_jit():
    """
    Feature: Compile cache.
    Description: Test compile cache in control flow with grad jit scenarios.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_compile_cache_bprop_grad_jit.py"
    run_twice_with_same_network(pypath, "./bprop_grad_jit", "bprop_grad_jit_first.txt",
                                "bprop_grad_jit_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_compile_cache_with_somas():
    """
    Feature: Compile cache without lazy inline.
    Description: Test compile cache when somas is enabled.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_with_somas.py"
    run_twice_with_same_network(pypath, "./net_with_somas", "net_with_somas_first.txt",
                                "net_with_somas_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_while_loop3():
    """
    Feature: Compile cache with control flow
    Description: Test compile cache when using WhileLoopEvaluator to handle ops.WhileLoop operation
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_while_loop.py"
    run_twice_with_same_network(pypath, "./net_while_loop", "net_while_loop_first.txt",
                                "net_while_loop_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_branch_same_shape():
    """
    Feature: Compile cache with control flow function.
    Description: Test compile cache when two branch must return the same shape.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_if_by_if.py"
    run_twice_with_same_network(pypath, "./net_if_by_if", "net_if_by_if_first.txt",
                                "net_if_by_if_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_grad_net_single_cache():
    """
    Feature: Compile cache with jit enable dynamic.
    Description: Test compile cache with enable_dynamic decorator with mutable tuple.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_with_jit_grad.py"
    run_twice_with_same_network(pypath, "./net_jit_grad", "net_jit_grad_first.txt", "net_jit_grad_second.txt",
                                "special")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_input_tuple_paramter_dynamic_jit():
    """
    Feature: Compile cache with jit enable dynamic.
    Description: Test compile cache with enable_dynamic decorator with mutable tuple.
                 The input is a tuple[param, cnode], and the cnode contains inputt parameters.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_enable_dynamic.py"
    run_twice_with_same_network(pypath, "./net_enable_dynamic", "net_enable_dynamic_first.txt",
                                "net_enable_dynamic_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dynamic_shape():
    """
    Feature: Compile cache with dynamic shape net.
    Description: Test compile cache with enable_dynamic decorator. Create a dynamic shape net get tensor.shape, and add.
                 The output nodes are nodified by backend, exported, but not loaded.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_input_signature_shape.py"
    run_twice_with_same_network(pypath, "./net_input_signature_shape", "net_input_signature_shape_first.txt",
                                "net_input_signature_shape_second.txt", "once")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_addn_pass_net():
    """
    Feature: Compile cache with addn pass.
    Description: Test compile cache with addn pass.
    Expectation: Run success.
    """
    fpath = os.path.realpath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compile_cache/run_net_addn_pass.py"
    run_twice_with_same_network(pypath, "./net_addn_pass", "net_addn_pass_first.txt",
                                "net_addn_pass_second.txt", "temp")
