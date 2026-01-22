# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
import re
import shutil
import subprocess
import pytest
import numpy as np
from mindspore import mutable, Tensor, nn, jit, ops
from mindspore.graph.api import ms_compile_cache
from mindspore import dtype as mstype
from tests.st.networks import utils
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

match_output = re.compile(r'AAA(.*?)BBB', re.S)
match_num = re.compile(r'\d+\.?\d*', re.S)


def exec_insert_command(regex, context, file_name):
    ret = os.system('sed -i "/{0}/{1}" {2}'.format(regex, context, file_name))
    if ret != 0:
        raise ValueError(
            'exec `sed -i "/{0}/{1}" {2}` failed.'.format(regex, context, file_name))
    return ret


def exec_cd_command(command):
    ret = os.system('cd "{0}"'.format(command))
    if ret != 0:
        raise ValueError('exec `cd  "{0}"` failed.'.format(command))
    return ret


def exec_cp_command(src, dst):
    ret = os.system("cp -af {0} {1}".format(src, dst))
    if ret != 0:
        raise ValueError("cp -af {0} {1}".format(src, dst))
    return ret


def exec_model_and_check_result(cur_model_path, dataset_path, config_path, cache_path, check_context):
    exec_shell = "export GLOG_v=2; export MS_COMPILER_CACHE_ENABLE=1; " \
                 + "export MS_COMPILER_CACHE_PATH={}; cd resnet/scripts; bash run_distribute_train.sh {} {} {}" \
        .format(cache_path, utils.rank_table_path, dataset_path, config_path)
    os.system(exec_shell)
    cmd = "ps -ef | grep python | grep train.py | grep -v grep"
    ret = utils.process_check(100, cmd)
    exec_shell = "unset MS_COMPILER_CACHE_ENABLE; unset MS_COMPILER_CACHE_PATH"
    os.system(exec_shell)
    assert ret
    log_file = os.path.join(cur_model_path, "scripts/train_parallel{}/log")
    for i in range(8):
        per_step_time = utils.get_perf_data(log_file.format(i))
        assert per_step_time < 60.0
    loss_list = []
    for i in range(8):
        loss = utils.get_loss_data_list(log_file.format(i))
        loss_list.append(loss[-1])
        with open(log_file.format(i), "r", encoding='utf-8') as f:
            data = f.read()
        assert check_context in data
        os.remove(log_file.format(i))
    loss = sum(loss_list) / len(loss_list)
    return loss


def run_twice_with_same_network(file_name, cache_path, log_file_name_first,
                                log_file_name_second, is_debug=False, run_time=1):
    # Clear compile cache folder and log files
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    if os.path.exists(log_file_name_first):
        os.remove(log_file_name_first)
    if os.path.exists(log_file_name_second):
        os.remove(log_file_name_second)
    assert not os.path.exists(cache_path)
    assert not os.path.exists(log_file_name_first)
    assert not os.path.exists(log_file_name_second)

    # First run without compile cache
    if not is_debug:
        cmd_first = "export GLOG_v=2; export MS_COMPILER_CACHE_ENABLE=1; " \
                    + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, file_name,
                                                                                     log_file_name_first)
    else:
        cmd_first = "export GLOG_v=0; export MS_COMPILER_CACHE_ENABLE=1; " \
                    + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, file_name,
                                                                                     log_file_name_first)
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r", encoding='utf-8') as f_first:
        data_first = f_first.read()
    if is_debug:
        print("\nmatch_output:\n", match_output, flush=True)
        print("\ndata_first:\n", data_first, flush=True)
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_first

    # Take out the result of the first run
    match_output_first = re.findall(match_output, data_first)
    assert len(match_output_first) == 2 * run_time
    array_first = []
    array_shape_first = []
    for i in range(run_time):
        nums_first = re.findall(match_num, match_output_first[2 * i])
        array_first.append(np.array([float(x) for x in nums_first]))
        shape_first = re.findall(match_num, match_output_first[2 * i + 1])
        array_shape_first.append(np.array([int(x) for x in shape_first]))

    # Second run with compile cache
    if not is_debug:
        cmd_second = "export GLOG_v=2; export MS_COMPILER_CACHE_ENABLE=1; " \
            + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, file_name,
                                                                             log_file_name_second)
    else:
        cmd_second = "export GLOG_v=0; export MS_COMPILER_CACHE_ENABLE=1; " \
            + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, file_name,
                                                                             log_file_name_second)
    subprocess.check_output(cmd_second, shell=True)
    assert os.path.exists(log_file_name_second)
    with open(log_file_name_second, "r", encoding='utf-8') as f_second:
        data_second = f_second.read()
    if is_debug:
        print("\ndata_second:\n", data_second, flush=True)

    has_log = "Use the compilation cache and execute the backend actions only. Be aware of correctness risks." in \
              data_second
    if not has_log:
        print(f'{data_second}')
    assert has_log

    # Take out the result of the second run
    match_output_second = re.findall(match_output, data_second)
    assert len(match_output_second) == 2 * run_time
    array_second = []
    array_shape_second = []
    for i in range(run_time):
        nums_second = re.findall(match_num, match_output_second[2 * i])
        array_second.append(np.array([float(x) for x in nums_second]))
        shape_second = re.findall(match_num, match_output_second[2 * i + 1])
        array_shape_second.append(np.array([int(x) for x in shape_second]))

    for i in range(run_time):
        assert np.allclose(array_first[i], array_second[i], 0.0001, 0.0001)
        assert (array_shape_first[i] == array_shape_second[i]).all()

    # Clean files
    os.remove(log_file_name_first)
    os.remove(log_file_name_second)
    shutil.rmtree(cache_path)


def run_twice_with_different_networks(file_name_first, file_name_second, cache_path, log_file_name_first,
                                      log_file_name_second):
    # Clear compile cache folder
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    assert not os.path.exists(cache_path)

    # First run without compile cache
    cmd_first = "export GLOG_v=2; export MS_COMPILER_CACHE_ENABLE=1; " \
                + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, file_name_first,
                                                                                 log_file_name_first)
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r", encoding='utf-8') as f_first:
        data_first = f_first.read()
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_first

    ge_cache = cache_path + "/rank_0/ge_cache"
    shutil.rmtree(ge_cache)

    # Second run with compile cache
    cmd_second = "export GLOG_v=2; export MS_COMPILER_CACHE_ENABLE=1; " \
                 + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, file_name_second,
                                                                                  log_file_name_second)
    subprocess.check_output(cmd_second, shell=True)
    assert os.path.exists(log_file_name_second)
    with open(log_file_name_second, "r", encoding='utf-8') as f_second:
        data_second = f_second.read()
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_second

    # Clean log files
    os.remove(log_file_name_first)
    os.remove(log_file_name_second)
    shutil.rmtree(cache_path)


def run_two_cells_networks_once(file_name, cache_path, log_file_name):
    # Clear compile cache folder
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    assert not os.path.exists(cache_path)

    # First run without compile cache
    cmd = "GLOG_v=2 MS_COMPILER_CACHE_ENABLE=1 MS_COMPILER_CACHE_PATH=" + cache_path + " python " + file_name \
          + " > " + log_file_name + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(log_file_name)
    assert os.path.exists(cache_path)
    with open(log_file_name, "r", encoding='utf-8') as f:
        data = f.read()
    assert data.count(
        "Check the consistency of dependency files hash failed. Execute all the compilation actions.") == 2

    # Clean log files
    os.remove(log_file_name)
    shutil.rmtree(cache_path)


def check_log(role, log_name, str_to_check):
    assert os.path.exists(role + "/" + log_name)
    with open(role + "/" + log_name, "r", encoding='utf-8') as f:
        data = f.read()
    assert str_to_check in data


def clear_and_make_run_dir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)
    assert not os.path.exists(dir_path)
    os.mkdir(dir_path)
    assert os.path.exists(dir_path)


def check_compile_cache_files(cache_path, role):
    assert os.path.exists(cache_path)
    assert os.path.exists(
        cache_path + "/rank_0/graph_cache/" + role + "compile_cache_0.mindir")
    assert os.path.exists(
        cache_path + "/rank_0/graph_cache/" + role + "compile_dependency.hash")


def run_network_once_with_force_use_compile_cache(file_name, cache_path, log_file_name_first):
    # Clear compile cache folder and log files
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    if os.path.exists(log_file_name_first):
        os.remove(log_file_name_first)
    assert not os.path.exists(cache_path)
    assert not os.path.exists(log_file_name_first)

    # First run without compile cache
    cmd_first = "export GLOG_v=2; export MS_DEV_FORCE_USE_COMPILE_CACHE=1; export MS_COMPILER_CACHE_ENABLE=1; " \
                + "export MS_COMPILER_CACHE_PATH={}; python {} > {} 2>&1".format(cache_path, file_name,
                                                                                 log_file_name_first)
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r", encoding='utf-8') as f_first:
        data_first = f_first.read()
    assert "The env MS_DEV_FORCE_USE_COMPILE_CACHE has been set. It will force to use the compile cache" in data_first
    assert "Failed to load the compilation cache file. Execute all the compilation actions." in data_first

    exec_shell = "unset MS_ENABLE_GE; unset MS_DEV_FORCE_USE_COMPILE_CACHE"
    os.system(exec_shell)

    # Clean files
    os.remove(log_file_name_first)
    shutil.rmtree(cache_path)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_compile_cache_load_weights():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache can load the value of parameters successfully.
    Expectation: success.
    """
    run_twice_with_same_network(
        "run_network_with_weights.py", "./weight", "weight_first.txt", "weight_second.txt")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_compile_cache_lenet():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function can run successfully.
    Expectation: success.
    """
    run_twice_with_same_network(
        "run_lenet.py", "./lenet", "lenet_first.txt", "lenet_second.txt", True)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@test_utils.run_test_with_On
def test_compile_cache_lenet_with_force_use_compile_cache():
    """
    Feature: Compile cache.
    Description: Test whether the env MS_DEV_FORCE_USE_COMPILE_CACHE takes effect.
    Expectation: success.
    """
    run_network_once_with_force_use_compile_cache("run_lenet.py", "./lenet_with_force_use_compile_cache",
                                                  "lenet_first.txt")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@test_utils.run_test_with_On
def test_compile_cache_net_with_control_flow():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache can load ref type parameter correctly.
    Expectation: success.
    """
    run_twice_with_same_network("run_network_with_control_flow.py", "./control_flow", "control_net_first.txt",
                                "control_net_second.txt")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_compile_cache_auto_detect():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache auto-detection function can run successfully.
    Expectation: success.
    """
    run_twice_with_different_networks("run_lenet.py", "run_network_with_weights.py", "./lenet_auto_detect",
                                      "auto_detect_first.txt", "auto_detect_second.txt")


@pytest.mark.skip
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_lenet_change_dir():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function can run successfully when changing
    the current work directory.
    Expectation: success.
    """
    cwd = os.getcwd()
    new_path = cwd + '/tmp'
    shutil.rmtree(new_path, ignore_errors=True)
    os.mkdir(new_path)
    os.chdir(new_path)
    run_twice_with_same_network("../run_lenet.py", "../lenet_change_dir", "../lenet_change_dir_first.txt",
                                "../lenet_change_dir_second.txt")
    os.chdir(cwd)
    shutil.rmtree(new_path, ignore_errors=True)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_ms_function():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache function can run successfully in the compilation of ms_function.
    Expectation: success.
    """
    run_twice_with_same_network("run_lenet_ms_function.py", "./lenet_ms_function", "lenet_ms_function_first.txt",
                                "lenet_ms_function_second.txt")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_run_two_cells_once():
    """
    Feature: Compile cache.
    Description: Test whether all the cells don't read the cached graph when run multiple cells once.
    Expectation: success.
    """
    run_two_cells_networks_once(
        "run_lenet_two_cells.py", "./lenet_two_cells", "lenet_two_cells.txt")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_resnet_infer_compile_cache():
    """
    Feature: Support compile cache in inference scenarios.
    Description: Support compile cache in inference scenarios.
    Expectation: Run success.
    """
    run_twice_with_same_network("run_resnet_infer.py", "./resnet_infer", "resnet_infer_first.txt",
                                "resnet_infer_second.txt")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_control_flow_partial_without_inputs():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache function can run successfully for the graph with a partial node
                 without inputs.
    Expectation: success.
    """
    run_twice_with_same_network("control_flow.py", "./control_flow_partial_without_inputs",
                                "control_flow_partial_without_inputs_first.txt",
                                "control_flow_partial_without_inputs_second.txt")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_with_inplace_tensor():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache function can run successfully for inplace feature.
    Expectation: success.
    """
    class TestNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assignadd = ops.AssignAdd()

        @jit(backend="ms_backend")
        def construct(self, kv_caches):
            k, v = kv_caches
            self.assignadd(k, ops.ones_like(k))
            self.assignadd(v, ops.ones_like(v))

    kv_cache_shape = (None, 1)
    kv_cache_dtype = mstype.int32
    dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
    dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
    dyn_kv_cache = mutable((dyn_key_cache, dyn_value_cache))

    model = TestNet()
    model.set_inputs(dyn_kv_cache)
    kv_cache_shape = (1, 1)
    key_cache = ops.ones(kv_cache_shape, dtype=kv_cache_dtype)
    value_cache = ops.ones(kv_cache_shape, dtype=kv_cache_dtype)
    kv_cache = mutable((key_cache, value_cache))

    model(kv_cache)
    assert len(ms_compile_cache) == 1
    assert kv_cache[0][0][0] == 2
    assert kv_cache[1][0][0] == 2

    model(kv_cache)
    assert len(ms_compile_cache) == 1
    assert kv_cache[0][0][0] == 3
    assert kv_cache[1][0][0] == 3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_grad_jit():
    """
    Feature: Compile cache.
    Description: Test compile cache with grad jit.
    Expectation: success.
    """
    run_twice_with_same_network("run_compile_cache_grad_jit.py", "./compile_cache_grad_jit",
                                "compile_cache_grad_jit_first.txt", "compile_cache_grad_jit_second.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_grad_jit_with_custom_bprop():
    """
    Feature: Compile cache.
    Description: Test compile cache with grad jit.
    Expectation: success.
    """
    run_twice_with_same_network("run_compile_cache_custom_bprop.py", "./compile_cache_custom_bprop",
                                "compile_cache_custom_bprop_first.txt", "compile_cache_custom_bprop.txt")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_compile_cache_lenet_with_jit():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function with jit can run successfully.
    Expectation: success.
    """
    run_twice_with_same_network(
        "run_lenet_with_jit.py", "./lenet_with_jit", "lenet_with_jit_first.txt", "lenet_with_jit_second.txt", False, 2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_high_grad_jit():
    """
    Feature: Compile cache.
    Description: Test compile cache with grad jit.
    Expectation: success.
    """
    run_twice_with_same_network("run_high_grad_jit.py", "./compile_cache_high_grad_jit",
                                "compile_cache_high_grad_jit_first.txt", "compile_cache_high_grad_jit_second.txt")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_jit_with_weight():
    """
    Feature: Compile cache.
    Description: Test compile cache with grad jit on net with weight.
    Expectation: success.
    """
    run_twice_with_same_network("run_jit_net_with_weight.py", "./compile_cache_jit_net_with_weight",
                                "compile_cache_jit_net_with_weight_first.txt",
                                "compile_cache_jit_net_with_weight_second.txt")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_jit_with_control_flow():
    """
    Feature: Compile cache.
    Description: Test compile cache with grad jit on net with control flow.
    Expectation: success.
    """
    run_twice_with_same_network("run_jit_with_control_flow.py", "./compile_cache_jit_with_control_flow",
                                "compile_cache_jit_with_control_flow_first.txt",
                                "compile_cache_jit_with_control_flow_second.txt")
