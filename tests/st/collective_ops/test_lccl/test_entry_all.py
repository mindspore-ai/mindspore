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
"""
The tests of mindspore, used to test LCCL communication ops.
"""

import os
import subprocess
from tests.mark_utils import arg_mark


kernel_allreduce = 'AllReduce'
kernel_allgather = 'AllGather'
kernel_reducescatter = 'ReduceScatter'
kernel_broadcast = 'Broadcast'
kernel_barrier = 'Barrier'
kernel_matmul_allreduce = 'MatMulAllReduce'
keyword_lccl = 'collective_comm_lib: "LCCL"'
keyword_hccl = 'collective_comm_lib: "HCCL"'


def check_keyword_in_ir(ir_path, kernel, keyword):
    cmd = f"grep '= {kernel}' {ir_path}"
    print(f'===== cmd: {cmd}', flush=True)
    result = subprocess.getoutput(cmd)
    return result.find(keyword) != -1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_lccl_allreduce():
    """
    Feature: lccl operator test.
    Description: msrun lccl all_reduce 8P case.
    Expectation: success
    """
    env = "export MS_ENABLE_LCCL=on;"
    env += "export MS_DEV_SAVE_GRAPHS=2;"
    env += "export MS_DEV_SAVE_GRAPHS_PATH=./graph_allreduce;"
    env += "export MS_DEV_DUMP_IR_PASSES=graph_build;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_lccl_allreduce.py"
    assert os.path.exists(script)
    msrun_path = real_path + "/output_allreduce"
    msrun_cmd = "msrun --worker_num=8 --local_worker_num=8 --join=True "\
                f"--log_dir={msrun_path} pytest -s {script}"

    return_code = os.system(f"{env} {msrun_cmd}")
    assert return_code == 0

    # Check if LCCL enabled.
    result = subprocess.getoutput("grep -rn 'Loading LCCL because' ./output_allreduce")
    assert result.find("Loading LCCL because env MS_ENABLE_LCCL is set to on") != -1
    # Check if LCCL kernel compiled.
    graph_path = './graph_allreduce/rank_0/graph_build_0_*'
    assert check_keyword_in_ir(graph_path, kernel_allreduce, keyword_lccl)
    assert not check_keyword_in_ir(graph_path, kernel_allreduce, keyword_hccl)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_lccl_allgather():
    """
    Feature: lccl operator test.
    Description: msrun lccl all_gather 8P case.
    Expectation: success
    """
    env = "export MS_ENABLE_LCCL=on;"
    env += "export MS_DEV_SAVE_GRAPHS=2;"
    env += "export MS_DEV_SAVE_GRAPHS_PATH=./graph_allgather;"
    env += "export MS_DEV_DUMP_IR_PASSES=graph_build;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_lccl_allgather.py"
    assert os.path.exists(script)
    msrun_path = real_path + "/output_allgather"
    msrun_cmd = "msrun --worker_num=8 --local_worker_num=8 --join=True "\
                f"--log_dir={msrun_path} pytest -s {script}"

    return_code = os.system(f"{env} {msrun_cmd}")
    assert return_code == 0

    # Check if LCCL enabled.
    result = subprocess.getoutput("grep -rn 'Loading LCCL because' ./output_allgather")
    assert result.find("Loading LCCL because env MS_ENABLE_LCCL is set to on") != -1
    # Check if LCCL kernel compiled.
    graph_path = './graph_allgather/rank_0/graph_build_0_*'
    assert check_keyword_in_ir(graph_path, kernel_allgather, keyword_lccl)
    assert not check_keyword_in_ir(graph_path, kernel_allgather, keyword_hccl)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_lccl_reducescatter():
    """
    Feature: lccl operator test.
    Description: msrun lccl reduce_scatter 8P case.
    Expectation: success
    """
    env = "export MS_ENABLE_LCCL=on;"
    env += "export MS_DEV_SAVE_GRAPHS=2;"
    env += "export MS_DEV_SAVE_GRAPHS_PATH=./graph_reducescatter;"
    env += "export MS_DEV_DUMP_IR_PASSES=graph_build;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_lccl_reduce_scatter.py"
    assert os.path.exists(script)
    msrun_path = real_path + "/output_reducescatter"
    msrun_cmd = "msrun --worker_num=8 --local_worker_num=8 --join=True "\
                f"--log_dir={msrun_path} pytest -s {script}"

    return_code = os.system(f"{env} {msrun_cmd}")
    assert return_code == 0

    # Check if LCCL enabled.
    result = subprocess.getoutput("grep -rn 'Loading LCCL because' ./output_reducescatter")
    assert result.find("Loading LCCL because env MS_ENABLE_LCCL is set to on") != -1
    # Check if LCCL kernel compiled.
    graph_path = './graph_reducescatter/rank_0/graph_build_0_*'
    assert check_keyword_in_ir(graph_path, kernel_reducescatter, keyword_lccl)
    assert not check_keyword_in_ir(graph_path, kernel_reducescatter, keyword_hccl)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_lccl_broadcast():
    """
    Feature: lccl operator test.
    Description: msrun lccl broadcast 8P case.
    Expectation: success
    """
    env = "export MS_ENABLE_LCCL=on;"
    env += "export MS_DEV_SAVE_GRAPHS=2;"
    env += "export MS_DEV_SAVE_GRAPHS_PATH=./graph_broadcast;"
    env += "export MS_DEV_DUMP_IR_PASSES=graph_build;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_lccl_broadcast.py"
    assert os.path.exists(script)
    msrun_path = real_path + "/output_broadcast"
    msrun_cmd = "msrun --worker_num=8 --local_worker_num=8 --join=True "\
                f"--log_dir={msrun_path} pytest -s {script}"

    return_code = os.system(f"{env} {msrun_cmd}")
    assert return_code == 0

    # Check if LCCL enabled.
    result = subprocess.getoutput("grep -rn 'Loading LCCL because' ./output_broadcast")
    assert result.find("Loading LCCL because env MS_ENABLE_LCCL is set to on") != -1
    # Check if LCCL kernel compiled.
    graph_path = './graph_broadcast/rank_0/graph_build_0_*'
    assert check_keyword_in_ir(graph_path, kernel_broadcast, keyword_lccl)
    assert not check_keyword_in_ir(graph_path, kernel_broadcast, keyword_hccl)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_lccl_barrier():
    """
    Feature: lccl operator test.
    Description: msrun lccl barrier 8P case.
    Expectation: success
    """
    env = "export MS_ENABLE_LCCL=on;"
    env += "export MS_DEV_SAVE_GRAPHS=2;"
    env += "export MS_DEV_SAVE_GRAPHS_PATH=./graph_barrier;"
    env += "export MS_DEV_DUMP_IR_PASSES=graph_build;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_lccl_barrier.py"
    assert os.path.exists(script)
    msrun_path = real_path + "/output_barrier"
    msrun_cmd = "msrun --worker_num=8 --local_worker_num=8 --join=True "\
                f"--log_dir={msrun_path} pytest -s {script}"

    return_code = os.system(f"{env} {msrun_cmd}")
    assert return_code == 0

    # Check if LCCL enabled.
    result = subprocess.getoutput("grep -rn 'Loading LCCL because' ./output_barrier")
    assert result.find("Loading LCCL because env MS_ENABLE_LCCL is set to on") != -1
    # Check if LCCL kernel compiled.
    graph_path = './graph_barrier/rank_0/graph_build_0_*'
    assert check_keyword_in_ir(graph_path, kernel_barrier, keyword_lccl)
    assert not check_keyword_in_ir(graph_path, kernel_barrier, keyword_hccl)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_lccl_matmul_allreduce():
    """
    Feature: lccl MatMulAllReduce fustion operator test.
    Description: lccl MatMulAllReduce 8P case.
    Expectation: success
    """
    env = "export MS_ENABLE_LCCL=on;"
    env += "export MS_DEV_SAVE_GRAPHS=2;"
    env += "export MS_DEV_SAVE_GRAPHS_PATH=./graph_matmul_allreduce;"
    env += "export MS_DEV_DUMP_IR_PASSES=graph_build;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_lccl_matmul_allreduce.py"
    assert os.path.exists(script)
    msrun_path = real_path + "/output_matmul_allreduce"
    msrun_cmd = "msrun --worker_num=8 --local_worker_num=8 --join=True "\
                f"--log_dir={msrun_path} pytest -s {script}"

    return_code = os.system(f"{env} {msrun_cmd}")
    assert return_code == 0

    # Check if LCCL enabled.
    result = subprocess.getoutput("grep -rn 'Loading LCCL because' ./output_matmul_allreduce")
    assert result.find("Loading LCCL because env MS_ENABLE_LCCL is set to on") != -1
    # Check if LCCL kernel compiled.
    graph_path = './graph_matmul_allreduce/rank_0/graph_build_1_*'
    assert check_keyword_in_ir(graph_path, kernel_matmul_allreduce, keyword_lccl)
    assert not check_keyword_in_ir(graph_path, kernel_matmul_allreduce, keyword_hccl)
