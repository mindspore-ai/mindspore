# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Utils for facade components."""

from . import keyword
from .config_util import get_function_config


def get_block_config():
    """
    Get Empty function config.
    """
    ret = {}
    ret[keyword.function] = []
    ret[keyword.inputs] = []
    ret[keyword.expect] = []
    return ret


def fill_block_config(ret, block_config, tid, group, desc_inputs, desc_bprop, expect,
                      desc_const, const_first, add_fake_input, fake_input_type):
    """
    Fill in block config.

    Args:
        ret (dict): The filled config.
        block_config (tuple): Block config.
        tid (str): Testing id.
        group (str): Testing group.
        desc_inputs (list): Inputs Description.
        desc_bprop (list): Backpropagation description.
        expect (list): Expectataion.
        desc_const (list): Const as inputs.
        const_first (bool): Const as first inputs.
        add_fake_input (bool): Add fake input.
        fake_input_type (numpy type): Type of faked input.

    Returns:
    """
    func_list = ret[keyword.function]
    inputs_list = ret[keyword.inputs]
    expect_list = ret[keyword.expect]

    block = block_config
    delta, max_error, input_selector, output_selector, \
    sampling_times, reduce_output, init_param_with, split_outputs, exception, error_keywords = get_function_config({})
    if isinstance(block_config, tuple) and isinstance(block_config[-1], dict):
        block = block_config[0]
        delta, max_error, input_selector, output_selector, \
        sampling_times, reduce_output, init_param_with, \
        split_outputs, exception, error_keywords = get_function_config(block_config[-1])

    if block is not None:
        func_list.append({
            keyword.id: tid,
            keyword.group: group,
            keyword.block: block,
            keyword.delta: delta,
            keyword.max_error: max_error,
            keyword.input_selector: input_selector,
            keyword.output_selector: output_selector,
            keyword.sampling_times: sampling_times,
            keyword.reduce_output: reduce_output,
            keyword.num_inputs: len(desc_inputs),
            keyword.num_outputs: len(desc_bprop),
            keyword.init_param_with: init_param_with,
            keyword.desc_const: desc_const,
            keyword.const_first: const_first,
            keyword.add_fake_input: add_fake_input,
            keyword.split_outputs: split_outputs,
            keyword.exception: exception,
            keyword.error_keywords: error_keywords
        })

    if desc_inputs or desc_const:
        inputs_list.append({
            keyword.id: tid,
            keyword.group: group,
            keyword.desc_inputs: desc_inputs,
            keyword.desc_bprop: desc_bprop,
            keyword.add_fake_input: add_fake_input,
            keyword.fake_input_type: fake_input_type
        })

    if expect:
        expect_list.append({
            keyword.id: tid + '-' + tid,
            keyword.group: group + '-' + group,
            keyword.desc_expect: expect
        })
