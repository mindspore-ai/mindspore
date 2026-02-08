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
1. copy mindformers to corresponding folder
2. replace parts of value in yaml through replace_config func
3. run st in dryrun mode
"""
import os
from tests.st.networks.llm_parallel_feature.utils import prepare_deepseekv3_testcase_env, check_log, DeepseekConfig, \
    log_path_preprocess

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_deepseekv3_cell_dp8mp2ep4pp2mb4gas1bs1_32p():
    """
    Feature: test deepseekv3 cell dp8mp2ep4pp2mb4gas1bs1 32p
    Description: test deepseekv3 cell dp8mp2ep4pp2mb4gas1bs1 32p
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp8mp2ep4pp2mb4gas1bs1_32p"
    rank_list = "8"
    mixtral_config = DeepseekConfig(case_name=case_name,
                                    num_layers=3,
                                    data_parallel=8,
                                    model_parallel=2,
                                    pipeline_stage=2,
                                    expert_parallel=4,
                                    micro_batch_num=4,
                                    enable_parallel_optimizer=True,
                                    vocab_emb_dp=True,
                                    offset=1,
                                    recompute=True)
    output_file, file_path = prepare_deepseekv3_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system("export MS_DEV_GRAPH_KERNEL_FLAGS='--enable_pass=grouped_matmul_assignadd_fusion'")
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 32 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
