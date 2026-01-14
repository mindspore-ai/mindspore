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
1. copy mindformers to corresponding folder
2. replace parts of value in yaml through replace_config func
3. run st in dryrun mode
"""
import os
from tests.st.networks.llm_parallel_feature.utils import prepare_testcase_env, check_log, LLMConfig, \
    log_path_preprocess, graph_path_preprocess, check_peak_memory, check_compile_time, check_graph, \
    check_param_shape, find_graph_file_name

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_qwen3_dp2mp4pp1_recompute():
    """
    Feature: test qwen3 dp2mp4pp1 full_recompute
    Description: test qwen3 dp2mp4pp1 full_recompute
    Expectation: st pass
    """
    case_name = "qwen3_dp2mp4pp1_recompute"
    rank_list = "0"
    qwen3_config = LLMConfig(case_name=case_name, data_parallel=2, model_parallel=4,
                             enable_parallel_optimizer=False, batch_size=4, vocab_emb_dp=False,
                             parallel_speed_up_json={
                                 'matmul_grad_comm_overlap': 'true'})
    output_file, file_path = prepare_testcase_env(case_name, qwen3_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name}")
    check_pair = {"Training Over": 1}
    real_graph_path = graph_path_preprocess(qwen3_config.save_graphs_path, rank_list)
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    graph_path = real_graph_path[0]
    attrs_check_pairs = {"recompute: Bool(1)": 27}
    validate_name = find_graph_file_name(graph_path, "validate")
    check_graph(graph_path, validate_name, attrs_check_pairs)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "41384")
        check_compile_time(log_path, 15)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only',
          essential_mark='essential')
def test_qwen3_dp4mp4pp1op_recompute():
    """
    Feature: test qwen3 dp4mp4pp1op full_recompute
    Description: test qwen3 dp4mp4pp1op full_recompute
    Expectation: st pass
    """
    case_name = "qwen3_dp4mp4pp1op_recompute"
    rank_list = "0"
    # wait for fixing
    qwen3_config = LLMConfig(case_name=case_name, data_parallel=4, model_parallel=4,
                             recompute=True, batch_size=2, vocab_emb_dp=False,
                             parallel_speed_up_json={
                                 'enable_grad_comm_opt': 'true',
                                 'enable_opt_shard_comm_opt': 'true'})

    output_file, file_path = prepare_testcase_env(case_name, qwen3_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name}")
    check_pair = {"Training Over": 1}
    real_graph_path = graph_path_preprocess(qwen3_config.save_graphs_path, rank_list)
    validate_name = find_graph_file_name(real_graph_path[0], "validate")
    graph_path = real_graph_path[0]
    attrs_check_pairs = {", recompute: Bool(1)": 345}
    check_graph(graph_path, validate_name, attrs_check_pairs)
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "15132")
        check_compile_time(log_path, 15)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only',
        essential_mark='essential', mem_peak=4.00)
def test_qwen3_cell_dp2mp4pp1op_grad_accu():
    """
    Feature: test qwen3 cell_dp2mp4pp1op_grad_accu
    Description: test qwen3 cell_dp2mp4pp1op_grad_accu
    Expectation: st pass
    """
    case_name = "qwen3_cell_dp2mp4pp1op_grad_accu"
    rank_list = "0"
    # wait for fixing
    qwen3_config = LLMConfig(case_name=case_name, data_parallel=2, model_parallel=4,
                             gradient_accumulation_steps=4, batch_size=1,
                             recompute=False,
                             parallel_speed_up_json={
                                 'enable_grad_comm_opt': 'false',
                                 'enable_opt_shard_comm_opt': 'false'})
    output_file, file_path = prepare_testcase_env(case_name, qwen3_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name} no_pp")
    check_pair = {"Training Over": 1}
    ops_check_pairs = {"VirtualAssignAdd": 43}
    graph_path = graph_path_preprocess(qwen3_config.save_graphs_path, rank_list)[0]
    step_parallel_end_name = find_graph_file_name(graph_path, "step_parallel_end")
    check_graph(graph_path, step_parallel_end_name, ops_check_pairs)
    param_opt_shape_check_pairs = {"_decoder.layers.0.self_attention.linear_qkv.weight": "(1280, 5120)",
                                   "accu_grads.decoder.layers.0.self_attention.linear_qkv.weight": "(2560, 5120)",
                                   "_adam_m.decoder.layers.0.self_attention.linear_qkv.weight": "(1280, 5120)"}
    check_param_shape(graph_path, step_parallel_end_name, 100, param_opt_shape_check_pairs)
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "26426")
        check_compile_time(log_path, 15)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_qwen3_cell_dp2mp4pp2vpp4op_1f1b():
    """
    Feature: test qwen3_cell_dp2mp4pp2vpp4op_1f1b
    Description: test qwen3_cell_dp2mp4pp2vpp4op_1f1b
    Expectation: st pass
    """
    case_name = "qwen3_cell_dp2mp4pp2vpp4op_1f1b"
    rank_list = "0,8"
    # wait for fixing
    qwen3_config = LLMConfig(case_name, data_parallel=2, model_parallel=4, pipeline_stage=2,
                             micro_batch_num=2, batch_size=2, pp_interleave_num=4,
                             pipeline_interleave=True, pipeline_scheduler="1f1b",
                             num_layers=8, recompute=False,
                             parallel_speed_up_json={
                                 'enable_grad_comm_opt': 'false',
                                 'enable_opt_shard_comm_opt': 'false'})
    output_file, file_path = prepare_testcase_env(case_name, qwen3_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_compile_time(log_path, 15)
    check_peak_memory(real_log_path[0], "20499")
    check_peak_memory(real_log_path[1], "27439")
    graph_path = graph_path_preprocess(qwen3_config.save_graphs_path, rank_list)
    # stage 0
    ops_check_pairs_0 = {"VirtualAssignAdd": 65}
    validate_name = find_graph_file_name(graph_path[0], "validate")
    step_parallel_end_name = find_graph_file_name(graph_path[0], "step_parallel_end")
    param_opt_shape_check_pairs = {"_decoder.layers.0.self_attention.linear_qkv.weight": "(1280, 5120)",
                                   "accu_grads.decoder.layers.0.self_attention.linear_qkv.weight": "(2560, 5120)",
                                   "_adam_m.decoder.layers.0.self_attention.linear_qkv.weight": "(1280, 5120)"}
    check_graph(graph_path[0], step_parallel_end_name, ops_check_pairs_0)
    check_param_shape(graph_path[0], validate_name, 100, param_opt_shape_check_pairs)
    # stage 1
    step_parallel_end_name_1 = find_graph_file_name(graph_path[1], "step_parallel_end")
    ops_check_pairs_1 = {"VirtualAssignAdd": 68}
    check_graph(graph_path[1], step_parallel_end_name_1, ops_check_pairs_1)
    check_param_shape(graph_path[0], validate_name, 100, param_opt_shape_check_pairs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_qwen3_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute():
    """
    Feature: test qwen3_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute
    Description: test qwen3_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute
    Expectation: st pass
    """
    case_name = "qwen3_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute"
    rank_list = "0,8"
    # wait for fixing
    qwen3_config = LLMConfig(case_name, data_parallel=2, model_parallel=1, pipeline_stage=2,
                             micro_batch_num=4, batch_size=1, pp_interleave_num=2,
                             pipeline_interleave=True, pipeline_scheduler="1f1b",
                             num_layers=4, context_parallel=4, select_recompute=True,
                             recompute=False, enable_parallel_optimizer=False,
                             parallel_speed_up_json={
                                 'enable_grad_comm_opt': 'true',
                                 'enable_opt_shard_comm_opt': 'true'})
    output_file, file_path = prepare_testcase_env(case_name, qwen3_config)
    graph_path = graph_path_preprocess(qwen3_config.save_graphs_path, rank_list)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    # stage0
    validate_name = find_graph_file_name(graph_path[0], "validate")
    attrs_check_pairs = {" recompute: Bool(1)": 4}
    check_graph(graph_path[0], validate_name, attrs_check_pairs)
    # stage1
    validate_name_1 = find_graph_file_name(graph_path[1], "validate")
    attrs_check_pairs_1 = {" recompute: Bool(1)": 4}
    check_graph(graph_path[1], validate_name_1, attrs_check_pairs_1)
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_compile_time(log_path, 15)
    check_peak_memory(real_log_path[0], "52719")
    check_peak_memory(real_log_path[1], "60558")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only',
          essential_mark='essential')
def test_qwen3_dp4mp4pp1op_recompute_2():
    """
    Feature: test qwen3 dp4mp4pp1op_recompute_2
    Description: test qwen3 dp4mp4pp1op_recompute_2
    Expectation: st pass
    """
    case_name = "qwen3_dp4mp4pp1op_recompute_2"
    rank_list = "0"
    qwen3_config = LLMConfig(case_name=case_name,
                             data_parallel=4,
                             model_parallel=4,
                             batch_size=2,
                             vocab_emb_dp=False,
                             optimizer_weight_shard_size=2,
                             parallel_speed_up_json={
                                 'enable_grad_comm_opt:': 'true',
                                 'enable_opt_shard_comm_opt:True': 'true'},
                             recompute=True)
    output_file, file_path = prepare_testcase_env(case_name, qwen3_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name}")
    real_graph_path = graph_path_preprocess(qwen3_config.save_graphs_path, rank_list)
    graph_path = real_graph_path[0]

    validate_name = find_graph_file_name(graph_path, "validate")
    step_parallel_end_name = find_graph_file_name(graph_path, "step_parallel_end")
    # op 权重切分
    param_opt_shape_check_pairs = {"_decoder.layers.0.self_attention.linear_qkv.weight": "(640, 5120)"}
    check_param_shape(graph_path, step_parallel_end_name, 100, param_opt_shape_check_pairs)

    # recompute
    attrs_check_pairs = {' recompute: Bool(1)': '345'}
    check_graph(graph_path, validate_name, attrs_check_pairs)

    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    log_path = None
    for log_path in real_log_path:
        check_log(log_path, check_pair)
    check_peak_memory(log_path, "15132")
    check_compile_time(log_path, 15)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_qwen3_dp8mp1pp1op():
    """
    Feature: test qwen3 dp8mp1pp1op
    Description: test qwen3 dp8mp1pp1op
    Expectation: st pass
    """
    case_name = "qwen3_dp8mp1pp1op"
    rank_list = "0"
    qwen3_config = LLMConfig(case_name=case_name,
                             data_parallel=8,
                             model_parallel=1,
                             batch_size=2)
    output_file, file_path = prepare_testcase_env(case_name, qwen3_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name}")

    real_graph_path = graph_path_preprocess(qwen3_config.save_graphs_path, rank_list)
    graph_path = real_graph_path[0]
    step_parallel_end_name = find_graph_file_name(graph_path, "step_parallel_end")
    # op 权重切分
    param_opt_shape_check_pairs = {"_decoder.layers.0.self_attention.linear_qkv.weight": "(1280, 5120)"}
    check_param_shape(graph_path, step_parallel_end_name, 100, param_opt_shape_check_pairs)

    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    log_path = None
    for log_path in real_log_path:
        check_log(log_path, check_pair)
    check_peak_memory(log_path, "61422")
    check_compile_time(log_path, 15)
