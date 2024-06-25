# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
from tests.st.networks import utils
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

match_output = re.compile(r'AAA(.*?)BBB', re.S)
match_num = re.compile(r'\d+\.?\d*', re.S)


def exec_insert_command(regex, context, file_name):
    ret = os.system('sed -i "/{0}/{1}" {2}'.format(regex, context, file_name))
    if ret != 0:
        raise ValueError('exec `sed -i "/{0}/{1}" {2}` failed.'.format(regex, context, file_name))
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
    exec_shell = f"export GLOG_v=2; export MS_COMPILER_CACHE_ENABLE=1; " \
                 + "export MS_COMPILER_CACHE_PATH={}; cd resnet/scripts; bash run_distribute_train.sh {} {} {}" \
                     .format(cache_path, utils.rank_table_path, dataset_path, config_path)
    os.system(exec_shell)
    cmd = "ps -ef | grep python | grep train.py | grep -v grep"
    ret = utils.process_check(100, cmd)
    exec_shell = f"unset MS_COMPILER_CACHE_ENABLE; unset MS_COMPILER_CACHE_PATH"
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
        with open(log_file.format(i), "r") as f:
            data = f.read()
        assert check_context in data
        os.remove(log_file.format(i))
    loss = sum(loss_list) / len(loss_list)
    return loss


def run_twice_with_same_network(file_name, cache_path, log_file_name_first, log_file_name_second, use_ge=False):
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
    cmd_first = f"export GLOG_v=2; python " + file_name + " '" + cache_path + "' > " + log_file_name_first + " 2>&1"
    if use_ge:
        cmd_first = f"export GLOG_v=2; python " + \
                    file_name + " '" + cache_path + "' > " + log_file_name_first + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r") as f_first:
        data_first = f_first.read()
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_first

    # Take out the result of the first run
    match_output_first = re.findall(match_output, data_first)
    assert len(match_output_first) == 2
    nums_first = re.findall(match_num, match_output_first[0])
    array_first = np.array([float(x) for x in nums_first])
    shape_first = re.findall(match_num, match_output_first[1])
    array_shape_first = np.array([int(x) for x in shape_first])

    # Second run with compile cache
    cmd_second = f"export GLOG_v=2; python " + file_name + " '" + cache_path + "' > " + log_file_name_second + " 2>&1"
    if use_ge:
        cmd_second = f"export GLOG_v=2; python " + \
                     file_name + " '" + cache_path + "' > " + log_file_name_second + " 2>&1"
    subprocess.check_output(cmd_second, shell=True)
    assert os.path.exists(log_file_name_second)
    with open(log_file_name_second, "r") as f_second:
        data_second = f_second.read()

    has_log = "Use the compilation cache and execute the backend actions only. Be aware of correctness risks." in \
              data_second
    if not has_log:
        print(f'{data_second}')
    assert has_log

    # Take out the result of the second run
    match_output_second = re.findall(match_output, data_second)
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
    shutil.rmtree(cache_path)


def run_compile_cache_mp(file_name, cache_path, log_file_name_first, log_file_name_second):
    # Clear compile cache folder and log files
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    assert not os.path.exists(cache_path)

    # First run without compile cache
    cmd = "bash run_compile_cache_mp.sh {} {} {} {}".format(file_name, cache_path, log_file_name_first,
                                                            utils.rank_table_path)
    os.system(cmd)
    check_cmd = "ps -ef | grep python | grep run_compile_cache_mp.py | grep -v grep"
    # wait for net train finish
    ret = utils.process_check(100, check_cmd)
    assert ret
    assert os.path.exists(cache_path)
    log_fullname = 'worker_0.log'
    assert os.path.exists(log_fullname)
    with open(log_fullname, "r") as f_first:
        data_first = f_first.read()
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_first
    for i in range(8):
        os.remove(f'worker_{i}.log')
    cmd = "bash run_compile_cache_mp.sh {} {} {} {}".format(file_name, cache_path, log_file_name_second,
                                                            utils.rank_table_path)
    os.system(cmd)
    ret = utils.process_check(100, check_cmd)
    assert ret

    log_fullname = 'worker_0.log'
    assert os.path.exists(log_fullname)
    with open(log_fullname, "r") as f_second:
        data_second = f_second.read()

    has_log = "Use the compilation cache and execute the backend actions only. Be aware of correctness risks." in \
              data_second
    if not has_log:
        print(f'{data_second}')
    assert has_log

    # Clean files
    shutil.rmtree(cache_path)
    for i in range(8):
        os.remove(f'worker_{i}.log')


def run_twice_with_different_networks(file_name_first, file_name_second, cache_path, log_file_name_first,
                                      log_file_name_second):
    # Clear compile cache folder
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    assert not os.path.exists(cache_path)

    # First run without compile cache
    cmd_first = f"GLOG_v=2 python " + file_name_first + " '" + cache_path + "' > " + log_file_name_first + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r") as f_first:
        data_first = f_first.read()
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_first

    ge_cache = cache_path + "/rank_0/ge_cache"
    shutil.rmtree(ge_cache)

    # Second run with compile cache
    cmd_second = f"GLOG_v=2 python " + file_name_second + " '" + cache_path + "' > " + log_file_name_second + " 2>&1"
    subprocess.check_output(cmd_second, shell=True)
    assert os.path.exists(log_file_name_second)
    with open(log_file_name_second, "r") as f_second:
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
    cmd = f"GLOG_v=2 python " + file_name + " '" + cache_path + "' > " + log_file_name + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(log_file_name)
    assert os.path.exists(cache_path)
    with open(log_file_name, "r") as f:
        data = f.read()
    assert data.count(
        "Check the consistency of dependency files hash failed. Execute all the compilation actions.") == 2

    # Clean log files
    os.remove(log_file_name)
    shutil.rmtree(cache_path)


def check_log(role, log_name, str_to_check):
    assert os.path.exists(role + "/" + log_name)
    with open(role + "/" + log_name, "r") as f:
        data = f.read()
    assert str_to_check in data


def start_ps_subprocess(script_path, cache_path, str_to_check, log_name):
    cwd = os.getcwd()
    # start sched first time.
    os.environ['MS_ROLE'] = 'MS_SCHED'
    cmd_first = f"cd " + cwd + "/sched && GLOG_v=2 python ../" + script_path + " ../" + cache_path + " > " \
                + log_name + " 2>&1"
    sched_process = subprocess.Popen(cmd_first, shell=True)
    # start server first time.
    os.environ['MS_ROLE'] = 'MS_PSERVER'
    cmd_first = f"cd " + cwd + "/server && GLOG_v=2 python ../" + script_path + " ../" + cache_path + " > " \
                + log_name + " 2>&1"
    server_process = subprocess.Popen(cmd_first, shell=True)
    # start worker first time.
    os.environ['MS_ROLE'] = 'MS_WORKER'
    cmd_first = f"cd " + cwd + "/worker && GLOG_v=2 python ../" + script_path + " ../" + cache_path + " > " \
                + log_name + " 2>&1"
    subprocess.run(cmd_first, shell=True, check=True)
    os.chdir(cwd)
    check_log("server", log_name, str_to_check)
    check_log("worker", log_name, str_to_check)
    sched_process.wait()
    server_process.wait()


def clear_and_make_run_dir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)
    assert not os.path.exists(dir_path)
    os.mkdir(dir_path)
    assert os.path.exists(dir_path)


def check_compile_cache_files(cache_path, role):
    assert os.path.exists(cache_path)
    assert os.path.exists(cache_path + "/rank_0/graph_cache/" + role + "compile_cache_0.mindir")
    assert os.path.exists(cache_path + "/rank_0/graph_cache/" + role + "compile_dependency.hash")


def run_lenet_ps_twice(file_name, cache_path, log_file_name_first, log_file_name_second):
    # Clear compile cache folder and log files
    shutil.rmtree(cache_path, ignore_errors=True)
    assert not os.path.exists(cache_path)
    clear_and_make_run_dir("sched")
    clear_and_make_run_dir("server")
    clear_and_make_run_dir("worker")
    # Set envs
    os.environ['MS_SCHED_HOST'] = '127.0.0.1'
    os.environ['MS_SCHED_PORT'] = '8182'
    os.environ['MS_SCHED_NUM'] = '1'
    os.environ['MS_SERVER_NUM'] = '1'
    os.environ['MS_WORKER_NUM'] = '1'
    # First run
    first_str_to_check = "Check the consistency of dependency files hash failed. Execute all the compilation actions."
    start_ps_subprocess(file_name, cache_path, first_str_to_check, log_file_name_first)
    assert os.path.exists(cache_path)
    check_compile_cache_files(cache_path, "MS_WORKER")
    check_compile_cache_files(cache_path, "MS_PSERVER")
    # Second run
    os.environ['MS_SCHED_PORT'] = '8183'
    second_str_to_check = "Use the compilation cache and execute the backend actions only. Be aware of correctness" \
                          " risks."
    start_ps_subprocess(file_name, cache_path, second_str_to_check, log_file_name_second)

    # Clear
    del os.environ['MS_SCHED_HOST']
    del os.environ['MS_SCHED_PORT']
    del os.environ['MS_ROLE']
    del os.environ['MS_SCHED_NUM']
    del os.environ['MS_SERVER_NUM']
    del os.environ['MS_WORKER_NUM']
    shutil.rmtree("sched", ignore_errors=True)
    shutil.rmtree("server", ignore_errors=True)
    shutil.rmtree("worker", ignore_errors=True)
    shutil.rmtree(cache_path, ignore_errors=True)


def run_network_once_with_force_use_compile_cache(file_name, cache_path, log_file_name_first, use_ge=False):
    # Clear compile cache folder and log files
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    if os.path.exists(log_file_name_first):
        os.remove(log_file_name_first)
    assert not os.path.exists(cache_path)
    assert not os.path.exists(log_file_name_first)

    # First run without compile cache
    cmd_first = f"export MS_DEV_FORCE_USE_COMPILE_CACHE=1; export GLOG_v=2; python " + file_name + " '" + \
                cache_path + "' > " + log_file_name_first + " 2>&1"
    if use_ge:
        cmd_first = f"export MS_DEV_FORCE_USE_COMPILE_CACHE=1; export GLOG_v=2; export MS_ENABLE_GE=1; python " + \
                    file_name + " '" + cache_path + "' > " + log_file_name_first + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r") as f_first:
        data_first = f_first.read()
    assert "The env MS_DEV_FORCE_USE_COMPILE_CACHE has been set. It will force to use the compile cache" in data_first
    assert "Failed to load the compilation cache file. Execute all the compilation actions." in data_first

    exec_shell = f"unset MS_ENABLE_GE; unset MS_DEV_FORCE_USE_COMPILE_CACHE"
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
    run_twice_with_same_network("run_network_with_weights.py", "./weight", "weight_first.txt", "weight_second.txt")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_compile_cache_lenet():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function can run successfully.
    Expectation: success.
    """
    run_twice_with_same_network("run_lenet.py", "./lenet", "lenet_first.txt", "lenet_second.txt")


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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
def test_compile_cache_lenet_ps():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function can run successfully with lenet in ps mode.
    Expectation: success.
    """
    run_lenet_ps_twice("run_lenet_ps.py", "./lenet_ps", "lenet_ps_first.txt", "lenet_ps_second.txt")


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
    run_two_cells_networks_once("run_lenet_two_cells.py", "./lenet_two_cells", "lenet_two_cells.txt")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_compile_cache_pipeline_parallel_and_recompute():
    """
    Feature: Compile cache.
    Description: Test whether pipeline parallel and recompute can successfullty with compile cache.
    Expectation: success.
    """
    run_compile_cache_mp("run_compile_cache_mp.py", "./pp_recompute", "pp_recompute_first",
                         "pp_recompute_second")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_lenet_ge():
    """
    Feature: Compile cache.
    Description: Test whether the ge compile cache function can run successfully.
    Expectation: success.
    """
    run_twice_with_same_network("run_lenet.py", "./lenet", "lenet_first.txt", "lenet_second.txt", True)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_resnet_infer_compile_cache():
    """
    Feature: Support compile cache in inference scenarios.
    Description: Support compile cache in inference scenarios.
    Expectation: Run success.
    """
    run_twice_with_same_network("run_resnet_infer.py", "./resnet_infer", "resnet_infer_first.txt",
                                "resnet_infer_second.txt")
