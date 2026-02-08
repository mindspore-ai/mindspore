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
"""Test utils for testing mindformers mcore dryrun"""
import re
import os
import shutil
import subprocess
import json


def replace_config(net_config, file_path):
    """replace qwen yaml by config in testcases"""
    old_list = [
        'enable_parallel_optimizer: False', 'vocab_emb_dp: True',
        'full_batch: False', 'num_hidden_layers: 8', 'gradient_accumulation_steps: 1', 'batch_size: 1',
        'batch_size: 6', 'micro_batch_num: 1', 'data_parallel: 8', 'model_parallel: 1', 'pipeline_stage: 1',
        'epochs: 2', 'recompute: True', 'select_recompute: False', 'use_seq_parallel: False',
        'offset: 0', "output_dir: './output'", 'save_graphs: False', 'save_graphs_path: "./graph"',
        "parallel_mode: 1"
    ]

    new_list = [
        f'enable_parallel_optimizer: {net_config.enable_parallel_optimizer}',
        f'vocab_emb_dp: {net_config.vocab_emb_dp}', f'full_batch: {net_config.full_batch}',
        f'num_hidden_layers: {net_config.num_layers}',
        f'gradient_accumulation_steps: {net_config.gradient_accumulation_steps}',
        f'batch_size: {net_config.batch_size}', f'batch_size: {net_config.batch_size}',
        f'micro_batch_num: {net_config.micro_batch_num}', f'data_parallel: {net_config.data_parallel}',
        f'model_parallel: {net_config.model_parallel}', f'pipeline_stage: {net_config.pipeline_stage}',
        f'epochs: {net_config.epochs}', f'recompute: {net_config.recompute}',
        f'select_recompute: {net_config.select_recompute}', f'use_seq_parallel: {net_config.use_seq_parallel}',
        f'offset: {net_config.offset}', f"output_dir: '{net_config.output_dir}'",
        f'save_graphs: {net_config.save_graphs}', f'save_graphs_path: "{net_config.save_graphs_path}"',
        f"parallel_mode: {net_config.parallel_mode}"
    ]

    if len(old_list) != len(new_list):
        print(f"Old list and new list have different lengths: {len(old_list)} and {len(new_list)}")
        return False
    for old, new in zip(old_list, new_list):
        if "'" in old:
            sed_cmd = f'''sed -i "s#{old}#{new}#g" {file_path}'''
        else:
            sed_cmd = f"sed -i 's#{old}#{new}#g' {file_path}"

        status, _ = subprocess.getstatusoutput(sed_cmd)
        if status != 0:
            print(f"Failed to replace {old} with {new} in {file_path}")
            return False

    # remove checkpoint monitor to prevent saving ckpt
    remove_checkpoint_monitor = f"sed -i '/CheckpointMonitor/,+4d' {file_path}"
    status, _ = subprocess.getstatusoutput(remove_checkpoint_monitor)
    if status != 0:
        print(f"Failed to remove CheckpointMonitor in {file_path}")
        return False
    # insert fine grain interleaved
    if net_config.fine_grain_interleave > 1:
        insert_fine_grain_interleave = r"sed -i '/model_config:/a\    fine_grain_interleave: {}' {}".format(
            net_config.fine_grain_interleave, file_path
        )
        status, _ = subprocess.getstatusoutput(insert_fine_grain_interleave)
        if status != 0:
            print(f"Failed to insert fine grain interleave in {file_path}")
    # insert context parallel
    if net_config.context_parallel:
        insert_context_parallel = r"sed -i '/vocab_emb_dp:/i\  context_parallel: {}' {}".format(
            net_config.context_parallel, file_path
        )
        status, _ = subprocess.getstatusoutput(insert_context_parallel)
        if status != 0:
            print(f"Failed to insert context parallel in {file_path}")
            return False
        # insert ring attention flag
        if net_config.use_ring_attention:
            insert_use_ring_attention = r"sed -i '/model_config:/a\    use_ring_attention: {}' {}".format(
                net_config.use_ring_attention, file_path
            )
            status, _ = subprocess.getstatusoutput(insert_use_ring_attention)
            if status != 0:
                print(f"Failed to insert use_ring_attention in {file_path}")
                return False

    # insert pipeline interleaved
    if net_config.pipeline_interleave:
        if net_config.pipeline_scheduler is None or net_config.pp_interleave_num == -1:
            print("pipeline_scheduler and pp_interleave_num should be set together")
            return False
        insert_pipeline_config = r"sed -i '/full_batch:/i\  pipeline_config:' {}".format(file_path)
        insert_pipeline_interleave = r"sed -i '/full_batch:/i\    pipeline_interleave: {}' {}".format(
            net_config.pipeline_interleave, file_path)
        insert_pipeline_scheduler = r"""sed -i '/full_batch:/i\    pipeline_scheduler: "{}"' {}""".format(
            net_config.pipeline_scheduler, file_path)
        insert_pp_interleave_num = r"""sed -i '/model_config:/a\    pp_interleave_num: {}' {}""".format(
            net_config.pp_interleave_num, file_path)
        for cmd in [insert_pipeline_config, insert_pipeline_interleave, insert_pipeline_scheduler,
                    insert_pp_interleave_num]:
            status, _ = subprocess.getstatusoutput(cmd)
            if status != 0:
                print(f"Failed to execute cmd {cmd} in {file_path}")
    # insert optimizer_weight_shard_size
    if net_config.optimizer_weight_shard_size != -1:
        insert_optimizer_weight_shard_size = (r"sed -i '/parallel_optimizer_config:/a\    optimizer_weight_shard_size: "
                                              r"{}' {}").format(net_config.optimizer_weight_shard_size, file_path)
        status, _ = subprocess.getstatusoutput(insert_optimizer_weight_shard_size)
        if status != 0:
            print(f"Failed to insert optimizer_weight_shard_size in {file_path}")

    return True


# 获取 XX.ir 图的文件名称，最大的那个文件
def find_graph_file_name(graph_path, file_name_keyword):
    """find the largest graph file, whose name is XX.ir"""
    largest_size = 0
    ir_graph_name = None

    for root, _, files in os.walk(graph_path):
        for file in files:
            if file.endswith('.ir') and file_name_keyword in file:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)

                if file_size > largest_size:
                    largest_size = file_size
                    ir_graph_name = file

    return ir_graph_name


def check_log(file_path, check_pairs=None):
    """check the number of key in check_pairs in log file is equal to the value"""
    log_error_count = subprocess.check_output(
        ["grep -rE '%s' %s | wc -l" % ("ERROR|Traceback", file_path)],
        shell=True)
    log_cnt = str(log_error_count, 'utf-8').strip()
    if log_cnt != "0":
        os.system(f"cat {file_path}")
    assert log_cnt == "0", f"Error found in {file_path}"
    if check_pairs is not None:
        for key_word, value in check_pairs.items():
            log_output = subprocess.check_output(
                ["grep -r '%s' %s | wc -l" % (key_word, file_path)],
                shell=True)
            log_cnt = str(log_output, 'utf-8').strip()
            assert log_cnt == str(value), (f"Failed to find {key_word} in {file_path} or content is not correct."
                                           f"Expected occurrences: {value}, but got {log_cnt}")


def check_graph(graph_path, graph_name, check_pairs):
    """check the number of key in check_pairs in graph file is equal to the value (string)"""
    file_path = os.path.join(graph_path, graph_name)
    if check_pairs is not None:
        for key_word, value in check_pairs.items():
            log_output = subprocess.check_output(
                ["grep -r '%s' %s | wc -l" % (key_word, file_path)],
                shell=True)
            log_cnt = str(log_output, 'utf-8').strip()
            assert log_cnt == str(value), (f"Failed to find {key_word} in {file_path} or content is not correct."
                                           f"Expected occurrences: {value}, but got {log_cnt}")


def check_peak_memory(file_path, expected_peak_memory):
    """check the peak memory in the file is equal to the value (string)"""
    peak_memory_output = subprocess.check_output(
        ["grep -r 'Actual peak memory usage (with fragments):' %s" % file_path],
        shell=True)
    peak_memory_output = str(peak_memory_output, 'utf-8').strip()
    peak_memory_value = re.findall(r"Actual peak memory usage \(with fragments\): (\d+)M", peak_memory_output)[0]
    assert int(peak_memory_value) <= int(expected_peak_memory), (
        f"Peak memory is not correct, expect less than {expected_peak_memory}, but got {peak_memory_value}")


def check_param_shape(graph_path, graph_name, param_lines, check_pairs):
    """check the shape of parameters (string) in the graph file"""
    file_path = os.path.join(graph_path, graph_name)
    if check_pairs is None:
        return
    for param_name, expected_shape in check_pairs.items():
        params_output = subprocess.check_output(
            [f"grep '# Params' {file_path} -A {param_lines} | grep {param_name}"],
            shell=True)
        split_params_output = str(params_output, 'utf-8').strip().split("\n")
        for line in split_params_output:
            real_shape = line.split("(")[1].strip().split(")")[0]
            real_shape = f"({real_shape})"
            assert real_shape == expected_shape, (f"The shape of {param_name} is not correct, expect {expected_shape}, "
                                                  f"but got {real_shape}")


def check_node_shape(graph_path, graph_name, check_pairs=None):
    """check_pairs = {'node_name': {'key_word1': {"input": [xxxx], "output": [xxxx]}}}"""
    file_path = os.path.join(graph_path, graph_name)
    if check_pairs is None:
        raise ValueError("check_pairs is None")
    for node_name, sub_dict in check_pairs.items():
        for key_word, expected_value in sub_dict.items():
            grep_output = subprocess.check_output(
                [
                    f"grep '{node_name}' {file_path} -A 1 | grep '{key_word}' -A 1 | awk '/{node_name}/{{f=1;next}} "
                    f"/--/{{f=0}} f'"],
                shell=True)
            if not grep_output:
                raise ValueError(f"Failed to find {node_name} in {file_path}")
            split_grep_output = str(grep_output, 'utf-8').strip().split("\n")
            for line in split_grep_output:
                in_out_shape = line.split(" -> ")
                input_shape = in_out_shape[0]
                output_shape = in_out_shape[1]
                regular_exp = r"NoShape|\(\d{1,5}(?:, \d{1,5}){0,5}\)|\(\)"
                all_input_shape = re.findall(regular_exp, input_shape)
                all_output_shape = re.findall(regular_exp, output_shape)
                input_expected_shape = expected_value.get("input", [])
                output_expected_shape = expected_value.get("output", [])
                for i, in_shape in enumerate(all_input_shape):
                    if in_shape == "-1":
                        continue
                    assert in_shape == input_expected_shape[i], (
                        f"The {i}th input shape of {node_name} is not correct, expect {input_expected_shape[i]}, "
                        f"but got {in_shape}")
                for i, out_shape in enumerate(all_output_shape):
                    if out_shape == "-1":
                        continue
                    assert out_shape == output_expected_shape[i], (
                        f"The {i}th output shape of {node_name} is not correct, expect {output_expected_shape[i]}, "
                        f"but got {out_shape}")


def check_node_strategy(graph_path, graph_name, check_pairs):
    """
    check the strategy (string) of nodes in the graph file
    check_pairs = {'node_name': {'key_word1': '((1,2,3),)', 'key_word2': 'strategy'}}
    """
    file_path = os.path.join(graph_path, graph_name)
    if check_pairs is None:
        raise ValueError("check_pairs is None")
    for node_name, sub_dict in check_pairs.items():
        for key_word, expected_value in sub_dict.items():
            grep_output = subprocess.check_output(
                [f"grep '{node_name}' {file_path} | grep '{key_word}'"],
                shell=True)
            if not grep_output:
                raise ValueError(f"Failed to find {node_name} in {file_path}")
            split_grep_output = str(grep_output, 'utf-8').strip().split("\n")
            for line in split_grep_output:
                real_value = re.findall(r".*in_strategy: (\(\(.*\)\))", line)[0]
                assert real_value in expected_value, (
                    f"The strategy of {node_name} is not correct, expect {expected_value}, but got {real_value}")


def check_node_dependency_backward_search(graph_path, graph_name, backward_lines, dependency_list):
    """dependency_list = [start_unique_node, use_node_idx, use_node_idx, ..., end_node_name]"""
    file_path = os.path.join(graph_path, graph_name)
    start_node = dependency_list.pop(0)
    matched_start_lines = subprocess.check_output(
        [f"grep '{start_node}' {file_path} -B {backward_lines}"],
        shell=True)
    if not matched_start_lines:
        raise ValueError(f"Failed to find {start_node} in {file_path}")
    split_matched_start_lines = str(matched_start_lines, 'utf-8').strip().split("\n")

    find_node_mark = re.findall(r"%\d+", split_matched_start_lines[-1])[0]
    for line in reversed(split_matched_start_lines):
        # find all %n mark in current node, remove the first one (self)
        current_node_mark = re.findall(r"%\d+", line)
        if not current_node_mark:
            continue
        # match the first %n mark in current node, continue to search its use_node
        if find_node_mark == current_node_mark[0]:
            all_use_node_mark = current_node_mark[1:]
            use_idx = dependency_list.pop(0)
            # The last value of dependency_list is the end node name
            if not dependency_list:
                # if the end node name is not in the last line, return False
                assert use_idx in line, f"Failed to find {use_idx} in {line}"
                break
            # find use node mark
            if use_idx >= len(all_use_node_mark):
                raise ValueError(f"{use_idx} is out of range, all_use_node_mark is {all_use_node_mark}")
            find_node_mark = all_use_node_mark[use_idx]
        else:
            continue
    if dependency_list:
        raise ValueError(f"Failed to find all dependency nodes in {file_path}")


def log_path_preprocess(log_file_name, rank_list, testcase_name):
    """return the log path list, combining with rank list"""
    log_path_list = []
    split_rank_list = rank_list.split(",")
    for rank in split_rank_list:
        log_path_list.append(f"./{testcase_name}/rank_{rank}/{log_file_name}")
    return log_path_list


def graph_path_preprocess(graph_path, rank_list):
    """return the graph path list, combining with rank list"""
    graph_path_list = []
    split_rank_list = rank_list.split(",")
    for rank in split_rank_list:
        graph_path_list.append(f"./{graph_path}/rank_{rank}/")
    return graph_path_list


# pylint: disable=W0703
def clear_directory(directory_path):
    """Delete all contents inside the specified directory."""
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def mock_third_party_pkg(pkg_name, file_path):
    """Mock third party package in the specified file."""
    mock_code = (
        f"""import sys\\nimport types\\nmock_pkg = types.ModuleType("{pkg_name}")\\nsys.modules["{pkg_name}"] = """
        f"mock_pkg")
    insert_code = f"sed -i '1i {mock_code}' {file_path}"
    status, _ = subprocess.getstatusoutput(insert_code)
    if status != 0:
        print(f"Failed to insert mock code to {file_path}")
        return False
    return True


def update_parallel_speed_up_json(case_name, net_config, yaml_path, deepseekv3=False):
    """Update parallel_speed_up_json"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    if deepseekv3:
        file_path = f"{sh_path}/deepseekv3/{case_name}/parallel_speed_up.json"

        # use json module:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(net_config.parallel_speed_up_json, f, indent=4)
    else:
        # 1. copy the parallel_speed_up.json to the testcase folder
        os.system(f"cp ./parallel_speed_up.json ./{case_name}")
        # 2. update the parallel_speed_up.json
        file_path = f"{sh_path}/{case_name}/parallel_speed_up.json"
        for key, value in net_config.parallel_speed_up_json.items():
            if isinstance(value, int):
                update_cmd = f"""sed -i 's#"{key}": 0#"{key}": {value}#g' {file_path}"""
            else:
                update_cmd = f"""sed -i 's#"{key}": false#"{key}": {value}#g' {file_path}"""
            status, _ = subprocess.getstatusoutput(update_cmd)
            if status != 0:
                print(f"Failed to update {key} to {value} in {file_path}")
                return False
    # 3. insert the parallel_speed_up.json to the yaml file
    insert_json = (r"""sed -i '/max_device_memory:/a\  ascend_config:\n    parallel_speed_up_json_path: "{}"' {}"""
                   .format(file_path, yaml_path))
    status, _ = subprocess.getstatusoutput(insert_json)
    if status != 0:
        print(f"Failed to insert parallel_speed_up.json to {yaml_path}")
        return False
    return True


def prepare_testcase_env(testcase_name, net_config):
    """Prepare testcase environment and files"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    # 1. create testcase folder
    os.makedirs(os.path.join(sh_path, testcase_name), exist_ok=True)
    # 2. clear folder (if exist)
    clear_directory(f"{sh_path}/{testcase_name}")
    # 3. copy yaml to testcase folder
    os.system(f"cp {sh_path}/pretrain_qwen3.yaml ./{testcase_name}")
    # 4. replace config in yaml
    file_path = f'{sh_path}/{testcase_name}/pretrain_qwen3.yaml'
    status = replace_config(net_config, file_path)
    run_mindformers_path = f'{sh_path}/../mindformers/run_mindformer.py'
    # 5. mock tiktoken
    mock_third_party_pkg("tiktoken", run_mindformers_path)
    # 6. update parallel_speed_up.json if needed
    if net_config.parallel_speed_up_json is not None:
        if not update_parallel_speed_up_json(testcase_name, net_config, file_path):
            raise ValueError("Failed to update parallel_speed_up.json")
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    output_file = f"./{testcase_name}_output.log"
    return output_file, file_path


def check_compile_time(log_file, percentage):
    """Check compile time"""
    keywords = ['pipeline_split', '.parallel', 'parallel_renormalize']
    result = find_sums_in_log(log_file, keywords)
    if result is None:
        return
    compile_time = 0
    for value in result:
        compile_time += float(value[:-1])
    compile_time = round(compile_time, 2)
    print('Parallel compilation time is : %s%%' % compile_time)
    assert compile_time <= percentage, "Compile time is too long!"


def find_sums_in_log(log_file, keywords):
    """Find sums in log file"""
    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            if 'Sums' in line:
                return find_keyword_in_next_lines(file, keywords)
    os.system(f"cat {log_file}")
    return None


def find_keyword_in_next_lines(file, keywords):
    """Find keyword in next lines"""
    result = []
    for line in file:
        for keyword in keywords:
            if keyword in line:
                result.append(line.split(":")[2].strip())
                print('Compile_time %s' % line)
                keywords.remove(keyword)
                break
    return result


def check_comm_op_groups(graph_path, graph_name, check_pairs):
    """
    check the group of comm_ops in the graph file
    check_pairs = {comm_ops_name: {key_words1: groups1, key_words2:groups2}}
    """
    file_path = os.path.join(graph_path, graph_name)
    if check_pairs is None:
        raise ValueError("check_pairs is None")
    for node_name, sub_dict in check_pairs.items():
        for key_word, expected_value in sub_dict.items():
            grep_output = subprocess.check_output(
                [f"grep '{node_name}' {file_path} | grep '{key_word}'"],
                shell=True)
            if not grep_output:
                raise ValueError(f"Failed to find {node_name} in {file_path}")
            split_grep_output = str(grep_output, 'utf-8').strip().split("\n")
            for line in split_grep_output:
                real_value = re.findall(r".*group_rank_ids: \((.*?)\)", line)[0]
                assert real_value in expected_value, (
                    f"The group_rank_ids of {node_name} is not correct, expect {expected_value}, but got {real_value}")


class LLMConfig:
    """add default config for LLM"""
    def __init__(self,
                 case_name,
                 enable_parallel_optimizer=True,
                 vocab_emb_dp=True,
                 full_batch=True,
                 num_layers=4,
                 gradient_accumulation_steps=1,
                 batch_size=1,
                 micro_batch_num=1,
                 data_parallel=8,
                 model_parallel=1,
                 pipeline_stage=1,
                 epochs=2,
                 sink_size=2,
                 recompute=False,
                 select_recompute=False,
                 use_seq_parallel=False,
                 offset=0,
                 output_dir="./output",
                 save_graphs=True,
                 fine_grain_interleave=1,
                 context_parallel=False,
                 pipeline_interleave=False,
                 pipeline_scheduler=None,
                 pp_interleave_num=-1,
                 parallel_speed_up_json=None,
                 optimizer_weight_shard_size=-1,
                 parallel_mode=1,
                 use_ring_attention=False):
        # output dir
        self.output_dir = output_dir

        # context
        self.save_graphs = save_graphs
        self.save_graphs_path = f"{case_name}/graphs"
        self.parallel_speed_up_json = parallel_speed_up_json

        # parallel context
        self.enable_parallel_optimizer = enable_parallel_optimizer
        self.full_batch = full_batch
        self.pipeline_interleave = pipeline_interleave
        self.pipeline_scheduler = pipeline_scheduler

        # parallel
        self.parallel_mode = parallel_mode
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.pipeline_stage = pipeline_stage
        self.context_parallel = context_parallel
        self.vocab_emb_dp = vocab_emb_dp
        self.micro_batch_num = micro_batch_num
        self.use_seq_parallel = use_seq_parallel
        self.optimizer_weight_shard_size = optimizer_weight_shard_size

        # model config
        self.num_layers = num_layers
        self.fine_grain_interleave = fine_grain_interleave
        self.offset = offset
        self.pp_interleave_num = pp_interleave_num
        self.use_ring_attention = use_ring_attention

        # runner config
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.sink_size = sink_size

        # recompute
        self.recompute = recompute
        self.select_recompute = select_recompute


class DeepseekConfig:
    """add default config for DeepSeek"""
    def __init__(self,
                 case_name,
                 enable_parallel_optimizer=True,
                 vocab_emb_dp=True,
                 full_batch=True,
                 num_layers=4,
                 gradient_accumulation_steps=1,
                 batch_size=1,
                 micro_batch_num=1,
                 data_parallel=8,
                 model_parallel=1,
                 pipeline_stage=1,
                 expert_parallel=1,
                 epochs=1,
                 sink_size=1,
                 recompute=False,
                 select_recompute=False,
                 use_seq_parallel=False,
                 offset=0,
                 output_dir="./output",
                 save_graphs=True,
                 fine_grain_interleave=1,
                 context_parallel=False,
                 pipeline_interleave=False,
                 pipeline_scheduler=None,
                 pp_interleave_num=-1,
                 parallel_speed_up_json=None,
                 optimizer_weight_shard_size=-1,
                 parallel_mode=1,
                 use_ring_attention=False,
                 group_wise_a2a=False,
                 jit_level="O1",
                 use_fused_ops_topkrouter=False,
                 npu_nums_per_device=8):
        # output dir
        self.output_dir = output_dir

        # context
        self.jit_level = jit_level
        self.save_graphs = save_graphs
        self.save_graphs_path = f"{case_name}/graphs"
        self.parallel_speed_up_json = parallel_speed_up_json

        # parallel context
        self.enable_parallel_optimizer = enable_parallel_optimizer
        self.full_batch = full_batch
        self.pipeline_interleave = pipeline_interleave
        self.pipeline_scheduler = pipeline_scheduler

        # parallel
        self.parallel_mode = parallel_mode
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.pipeline_stage = pipeline_stage
        self.expert_parallel = expert_parallel
        self.context_parallel = context_parallel
        self.vocab_emb_dp = vocab_emb_dp
        self.micro_batch_num = micro_batch_num
        self.use_seq_parallel = use_seq_parallel
        self.optimizer_weight_shard_size = optimizer_weight_shard_size

        # model config
        self.num_layers = num_layers
        self.fine_grain_interleave = fine_grain_interleave
        self.offset = offset
        self.pp_interleave_num = pp_interleave_num
        self.use_ring_attention = use_ring_attention

        # runner config
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.sink_size = sink_size

        # recompute
        self.recompute = recompute
        self.select_recompute = select_recompute

        # moe config
        self.group_wise_a2a = group_wise_a2a
        self.use_fused_ops_topkrouter = use_fused_ops_topkrouter
        self.npu_nums_per_device = npu_nums_per_device


def prepare_deepseekv3_testcase_env(testcase_name, net_config):
    """Prepare DeepSeekv3 testcase environment and files"""
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    # 1. create testcase folder
    os.makedirs(os.path.join(sh_path, testcase_name), exist_ok=True)
    # 2. clear folder (if exist)
    clear_directory(f"{sh_path}/{testcase_name}")
    # 3. copy yaml to testcase folder
    os.system(f"cp {sh_path}/pretrain_deepseek3.yaml ./{testcase_name}")
    # 4. replace config in yaml
    file_path = f'{sh_path}/{testcase_name}/pretrain_deepseek3.yaml'
    status = replace_deepseekv3_config(net_config, file_path)
    run_mindformers_path = f'{sh_path}/../mindformers/run_mindformer.py'
    # 5. mock tiktoken
    mock_third_party_pkg("tiktoken", run_mindformers_path)
    # 6. update parallel_speed_up.json if needed
    if net_config.parallel_speed_up_json is not None:
        if not update_parallel_speed_up_json(testcase_name, net_config, file_path):
            raise ValueError("Failed to update parallel_speed_up.json")
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    output_file = f"./{testcase_name}_output.log"
    return output_file, file_path


def replace_deepseekv3_config(net_config, file_path):
    """Replace DeepSeekv3 config"""
    old_list = [
        'enable_parallel_optimizer: True', 'vocab_emb_dp: True',
        'full_batch: True', 'num_layers 61', 'micro_batch_num 2',
        'data_parallel: 2', 'model_parallel: 2', 'pipeline_stage: 2', 'expert_parallel: 2', 'recompute: True',
        'select_recompute: False', 'offset: 0', "npu_nums_per_device: 8",
        'save_graphs: False', 'save_graphs_path: "./graph"'
    ]

    new_list = [
        f'enable_parallel_optimizer: {net_config.enable_parallel_optimizer}',
        f'vocab_emb_dp: {net_config.vocab_emb_dp}', f'full_batch: {net_config.full_batch}',
        f'num_layers {net_config.num_layers}',
        f'micro_batch_num {net_config.micro_batch_num}',
        f'data_parallel: {net_config.data_parallel}',
        f'model_parallel: {net_config.model_parallel}',
        f'pipeline_stage: {net_config.pipeline_stage}',
        f'expert_parallel: {net_config.expert_parallel}',
        f'recompute: {net_config.recompute}',
        f'select_recompute: {net_config.select_recompute}',
        f'offset: {net_config.offset}',
        f'npu_nums_per_device: {net_config.npu_nums_per_device}',
        f'save_graphs: {net_config.save_graphs}', f'save_graphs_path: "{net_config.save_graphs_path}"'
    ]

    if len(old_list) != len(new_list):
        print(f"Old list and new list have different lengths: {len(old_list)} and {len(new_list)}")
        return False
    for old, new in zip(old_list, new_list):
        if "'" in old:
            sed_cmd = f'''sed -i "s#{old}#{new}#g" {file_path}'''
        else:
            sed_cmd = f"sed -i 's#{old}#{new}#g' {file_path}"

        status, _ = subprocess.getstatusoutput(sed_cmd)
        if status != 0:
            print(f"Failed to replace {old} with {new} in {file_path}")
            return False

    # insert gradient accumulation steps
    if net_config.gradient_accumulation_steps:
        insert_gradient_accumulation_steps = r"sed -i '/runner_config:/a\  gradient_accumulation_steps: {}' {}".format(
            net_config.gradient_accumulation_steps, file_path
        )
        status, _ = subprocess.getstatusoutput(insert_gradient_accumulation_steps)
        if status != 0:
            print(f"Failed to insert gradient_accumulation_steps in {file_path}")

    # insert pipeline interleaved
    if net_config.pipeline_interleave:
        if net_config.pipeline_scheduler is None or net_config.pp_interleave_num == -1:
            print("pipeline_scheduler and pp_interleave_num should be set together")
            return False
        insert_pipeline_config = r"sed -i '/full_batch:/i\  pipeline_config:' {}".format(file_path)
        insert_pipeline_interleave = r"sed -i '/full_batch:/i\    pipeline_interleave: {}' {}".format(
            net_config.pipeline_interleave, file_path)
        insert_pipeline_scheduler = r"""sed -i '/full_batch:/i\    pipeline_scheduler: "{}"' {}""".format(
            net_config.pipeline_scheduler, file_path)
        insert_pp_interleave_num = r"""sed -i '/model_config:/a\    pp_interleave_num: {}' {}""".format(
            net_config.pp_interleave_num, file_path)
        for cmd in [insert_pipeline_config, insert_pipeline_interleave, insert_pipeline_scheduler,
                    insert_pp_interleave_num]:
            status, _ = subprocess.getstatusoutput(cmd)
            if status != 0:
                print(f"Failed to execute cmd {cmd} in {file_path}")
    return True
