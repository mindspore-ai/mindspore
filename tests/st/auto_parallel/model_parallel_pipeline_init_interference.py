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
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_mp_2_1_1_2_pp():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_logs/strategy_2_1_1_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_autoparallel_pipeline_config")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_mp_2_1_1_2_pp_lazy_init():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_logs/strategy_2_1_1_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_pp_config_lazy_init")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_mp_2_1_1_2_pp_lazy_init_dp():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_logs/strategy_2_1_1_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_pp_config_lazy_init_dp")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_mp_2_1_1_2_pp_lazy_init_inline_dp():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_logs/strategy_2_1_1_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_pp_config_with_lazy_init_inline")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_mp_2_1_1_2_pp_lazy_init_inline_sink():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: The error of the loss is within the allowable range.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_functional_programming_logs/strategy_2_1_1_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_pp_cfg_lazy_init_inline_sink")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_mp_1_2_2_2_functional_programming():
    '''
    Feature: Model parallel, strategy:((1,1),(1,2)).
    Description: Test model parallel.
    Expectation: The error of the loss is within the allowable range.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_functional_programming_logs/strategy_1_1_1_2 pytest -s -v "
                    "model_parallel_pipeline_init.py::test_parallel_mp_compare_context_auto_fun_programming")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_mp_1_2_2_2_model_programming():
    '''
    Feature: Model parallel, strategy:((1,2),(2,2)).
    Description: Test model parallel.
    Expectation: The error of the loss is within the allowable range.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_functional_programming_logs/strategy_1_2_2_2 pytest -s -v "
                    "model_parallel_pipeline_init.py::test_parallel_mp_compare_context_model")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_mp_1_2_2_2_shared_params():
    '''
    Feature: Model parallel, strategy:((1,2),(2,2)).
    Description: Test model parallel.
    Expectation: The error of the loss is within the allowable range.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_functional_programming_logs/strategy_1_2_2_2 pytest -s -v "
                    "pipeline_inference.py::test_pipeline_inference_shared_params")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_parallel_pp_cfg_lazy_init_inline_sink_gpipe():
    '''
    Feature: Model parallel, strategy:((1,2),(2,2)).
    Description: Test model parallel.
    Expectation: The error of the loss is within the allowable range.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_functional_programming_logs/strategy_1_2_2_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_sink_gpipe")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_parallel_pp_cfg_lazy_init_inline_sink_seqpipe():
    '''
    Feature: Model parallel, strategy:((1,2),(2,2)).
    Description: Test model parallel.
    Expectation: The error of the loss is within the allowable range.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_functional_programming_logs/strategy_1_2_2_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_sink_seqpipe")
    assert ret == 0
