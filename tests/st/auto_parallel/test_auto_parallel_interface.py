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

"""test auto parallel cases"""
import os
import locale
import shutil
import subprocess
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_msrun_pipeline_remove_redundancy():
    '''
    Feature: test remove redundancy with auto_parallel interface.
    Description: test pipeline net train and predict using msrun.
    Expectation: run success.
    '''
    dir_to_remove = "./test_cpkt_pp2"
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10901 --join=True --log_dir=./test_cpkt_pp2/msrun_log "
        "pytest -s cpkt_rm_redundancy_auto_parallel.py::test_cpkt_remove_redundancy_precision"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_msrun_stand_alone_remove_redundancy():
    '''
    Feature: test remove redundancy with auto_parallel interface.
    Description: test stand alone.
    Expectation: run success.
    '''
    dir_to_remove = "./test_cpkt_stand_alone"
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10901 --join=True --log_dir=./test_cpkt_stand_alone/msrun_log "
        "pytest -s cpkt_rm_redundancy_auto_parallel.py::test_stand_alone_remove_redundancy"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_msrun_cpkt_transfer_functional():
    '''
    Feature: test checkpoints file transfer with auto_parallel interface.
    Description: test checkpoints file and strategy transfer with model or functional programming using msrun.
    Expectation: run success.
    '''
    exec_path = os.path.dirname(os.path.realpath(__file__))
    dir_to_remove = os.path.join(exec_path, "test_cpkt_transfer")
    log_dir = "./test_cpkt_transfer"
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10902 --join=True --log_dir=./test_cpkt_transfer/msrun_log "
        "pytest -s cpkt_transfer_functional_model_auto_parallel.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_msrun_transformer_opt():
    '''
    Feature: test transformer_opt in auto_parallel interface.
    Description: test ascend config set by transformer_opt using msrun.
    Expectation: run success.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10903 --join=True --log_dir=./test_ascend_config/ascend_config_log "
        "pytest -s transformer_opt_auto_parallel.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_config():
    '''
    Feature: test hccl buffer size in auto_parallel interface.
    Description: control buffer size by MS_DEV_HCCL_CONF using msrun.
    Expectation: run success.
    '''
    os.environ['GLOG_v'] = str(1)
    os.environ['MS_DEV_HCCL_CONF'] = (
        "enable_hccl_config:True,"
        "hccl_customized_default:100MB,"
        "hccl_list_config:0-1-2-3=150MB|4-5-6-7=50MB,"
        "hccl_stride_config:0-2:2=80MB|5-7:2=120MB"
    )

    command = (
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10801 --join=True --log_dir=./test_hccl_config/msrun_log "
        "pytest -s hccl_config.py::test_hccl_config"
    )

    output_lines = []
    with subprocess.Popen(command, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding=locale.getpreferredencoding(False), errors='ignore') as proc:
        for line in proc.stdout:
            print(line, end='')
            output_lines.append(line)

    full_output = ''.join(output_lines)
    assert "customized hcclBufferSize: 100 MB" in full_output, "Expected buffer size not found in log."
    assert "customized hcclBufferSize: 150 MB" in full_output, "Expected buffer size not found in log."
    assert "customized hcclBufferSize: 80 MB" in full_output, "Expected buffer size not found in log."

    del os.environ['MS_DEV_HCCL_CONF']


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_union_hccl_create():
    '''
    Feature: test union hccl groups creation in auto_parallel interface.
    Description: union hccl groups creation by python and C++ using msrun.
    Expectation: run success.
    '''
    os.environ['GLOG_v'] = str(1)
    dir_to_remove = "./test_union_hccl_groups"
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)

    command = (
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10905 --join=True --log_dir=./test_union_hccl_groups/msrun_log "
        "pytest -s hccl_config.py::test_union_hccl_groups"
    )

    output_lines = []
    with subprocess.Popen(command, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding=locale.getpreferredencoding(False), errors='ignore') as proc:
        for line in proc.stdout:
            print(line, end='')
            output_lines.append(line)

    full_output = ''.join(output_lines)
    assert "The group 'customed groups 0-1, 0' has been created, the ranks are: 0-1" not in full_output
    assert "The group 'customed groups 0-1, 1' has been created, the ranks are: 0-1" in full_output
    assert "The group 'customed groups 0-1, 2' has been created, the ranks are: 0-1" not in full_output
    assert "The group 'customed groups 0-1, 3' has been created, the ranks are: 0-1" in full_output
