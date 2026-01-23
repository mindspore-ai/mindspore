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
Tests for operation.
"""
import copy
import pytest
import numpy as np
import mindspore as ms
from mindspore._c_expression import MSContext
from tests.st.common.multi_process_actuator import run_cases_multi_process, get_max_worker
from tests.mark_utils import arg_mark
from tests.st.ops import (
    test_dynamic_quant,
    test_func_conv2d,
    test_matmul_conv_hf32,
    test_mint_avg_pool2d,
    test_mint_conv_transpose2d,
    test_mint_isinf,
    test_mint_isneginf,
    test_mint_max_pool2d,
    test_mint_nansum,
    test_moeffn,
    test_nn_conv2d,
    test_ops_adaptive_avg_pool1d,
    test_ops_add_layer_norm,
    test_ops_add_rms_norm,
    test_ops_apply_rotary_pos_emb,
    test_ops_concat,
    test_ops_conv2d,
    test_ops_conv3d_ext,
    test_ops_dropout_ext,
    test_ops_expm1,
    test_ops_flash_attention_score,
    test_ops_speed_fusion_attention,
    test_ops_fold,
    test_ops_gcd,
    test_ops_gelu,
    test_ops_generate_eod_mask_v2,
    test_ops_hardswish,
    test_ops_histc_ext,
    test_ops_index_fill_scalar,
    test_ops_index_fill_tensor,
    test_ops_index,
    test_ops_inplace_addmm,
    test_ops_inplace_index_put,
    test_ops_prompt_flash_attention,
    test_ops_rms_norm,
    test_ops_rotary_position_embedding,
    test_ops_rotated_iou,
    test_ops_smooth_l1_loss,
    test_ops_squeeze,
    test_ops_svd,
    test_ops_swiglu,
    test_ops_transpose,
    test_ops_unbind,
    test_ops_unfold,
    test_ops_upsample_nearest,
    test_ops_var,
    test_binary_ops,
    test_unary_ops,
    test_other_ops,
    test_reduction_ops,
)
from tests.st.mint import (
    test_addmv,
    test_cdist,
    test_diff,
    test_float_power,
    test_multinomial,
    test_mint_nan_to_num,
    test_nn_linear,
    test_nn_AvgPool3d,
    test_nn_kldivloss,
    test_matmul,
)

all_platform = ("ascend", "ascend910b", "cpu", "gpu")

def op_db_group_cases_register(model, func, platform=all_platform, alloc_mem=16):
    return [func,
            platform,
            alloc_mem,
            model.ops_params_dict[func.__name__]]

ops_group_cases_registry_level0 = {
    # '''
    # cases in this registry will be running as level0 group cases on the gate
    # args:
    #     case_func(python function): case function to run
    #     platform(tuple(str)): use ascend or ascend910b
    #     memory(int): memory will be used during case running
    #     parameter(tuple(tuple(any))): a parameter list for cases, support muti-parameters
    # '''
    'ops': [
        [test_func_conv2d.test_conv2d_binary_cases,
         ("ascend910b",), 246, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_func_conv2d.test_ops_conv2d_padding_same,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_matmul_conv_hf32.test_hf32,
         ("ascend910b",), 4, (("KBK", "PYBOOST"),)],
        [test_mint_avg_pool2d.test_avg_pool2d_and_double_backward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_mint_conv_transpose2d.test_conv_transpose2d_hf32,
         ("ascend910b",), 4, (("KBK", "PYBOOST"),)],
        [test_mint_isinf.test_net_2d_float32,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_mint_isneginf.test_net_2d_float32,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_mint_max_pool2d.test_ops_max_pool2d_forward_return_indices,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_mint_max_pool2d.test_ops_max_pool2d_forward_without_return_indices,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_mint_nansum.test_mint_nansum,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_moeffn.test_ffn_forward_net,
         ("ascend910b",), 16, None],
        [test_moeffn.test_ffn_forward_mode,
         ("ascend910b",), 16, (('KBK', 'pynative'),)],
        [test_nn_conv2d.test_nn_conv2d_default,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_nn_conv2d.test_nn_conv2d_padding_same,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_adaptive_avg_pool1d.test_adaptive_avg_pool1d,
         ("ascend910b",), 32, (('pynative', 'KBK'),)],
        [test_ops_add_layer_norm.test_add_layer_norm,
         ("ascend910b",), 72, ((ms.float32, ms.float16, ms.bfloat16), (ms.GRAPH_MODE, ms.PYNATIVE_MODE),
         (True, False))],
        [test_ops_add_rms_norm.test_add_rms_norm_forward_backward,
         ("ascend910b",), 32, (('pynative', 'KBK'),)],
        [test_ops_apply_rotary_pos_emb.test_apply_rotary_pos_emb_case0,
         ("ascend910b",), 86, (('GE',),)],
        [test_ops_conv2d.test_conv2d_forward,
         ("ascend910b",), 8, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_conv3d_ext.test_ops_conv3d_default,
         ("ascend910b",), 32, ((ms.PYNATIVE_MODE, ms.GRAPH_MODE),)],
        [test_ops_conv3d_ext.test_ops_conv3d_batchfy,
         ("ascend910b",), 32, ((ms.PYNATIVE_MODE, ms.GRAPH_MODE),)],
        [test_ops_dropout_ext.test_func_dropout_ext_normal,
         ("ascend910b",), 512, (('kbk', 'pynative'), (True, False))],
        [test_ops_dropout_ext.test_nn_dropout_ext_normal,
         ("ascend910b",), 512, (('kbk', 'pynative'), (True, False))],
        [test_ops_expm1.test_ops_expm1_normal,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_flash_attention_score.test_ops_flash_attention_score,
         ("ascend910b",), 36, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (ms.float16, ms.bfloat16))],
        [test_ops_speed_fusion_attention.test_speed_fusion_attention_normal,
         ("ascend910b",), 38, ((ms.PYNATIVE_MODE,),)],
        [test_ops_fold.test_fold,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_gcd.test_ops_gcd_binary_cases,
         ("ascend910b",), 26, (("pynative", "KBK", "GRAPH"),)],
        [test_ops_generate_eod_mask_v2.test_generate_eod_mask_v2,
         ("ascend910b",), 62, (("KBK", "PYBOOST"),)],
        [test_ops_hardswish.test_ops_hardswish_normal,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_histc_ext.test_ops_histc_ext_normal,
         ("ascend910b",), 36, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_index_fill_scalar.test_ops_index_fill_scalar,
         ("ascend910b",), 4, (("pynative", "KBK"),)],
        [test_ops_index_fill_tensor.test_ops_index_fill_tensor,
         ("ascend910b",), 4, (("pynative", "KBK"),)],
        [test_ops_index.test_ops_index_forward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_index.test_ops_index_backward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_inplace_addmm.test_inplace_addmm,
         ("ascend910b",), 4, ((ms.float32, ms.float16),)],
        [test_ops_inplace_index_put.test_inplace_index_put_forward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_inplace_index_put.test_inplace_index_put_backward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_prompt_flash_attention.test_ops_prompt_flash_attention_normal,
         ("ascend910b",), 38, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (ms.float16, ms.bfloat16), ("BSH", "BNSD"))],
        [test_ops_rms_norm.test_rms_norm_forward,
         ("ascend910b",), 16, (('pynative', 'KBK'), (np.float32, np.float16))],
        [test_ops_rotary_position_embedding.test_ops_rotary_position_embedding,
         ("ascend910b",), 36, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (ms.float32, ms.float16, ms.bfloat16), (0, 1))],
        [test_ops_rotated_iou.test_ops_rotated_iou_binary_cases,
         ("ascend910b",), 4, ((ms.PYNATIVE_MODE,),)],
        [test_ops_smooth_l1_loss.test_ops_smooth_l1_loss_normal,
         ("ascend910b",), 4, (("pynative", "KBK", "graph"), ("mean", "sum", "none"))],
        [test_ops_squeeze.test_ops_squeeze_normal,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_svd.test_ops_svd_normal,
         ("ascend910b",), 4, (('pynative', 'kbk', 'ge'), (np.float32, np.float64, np.complex64, np.complex128))],
        [test_ops_svd.test_ops_svd_selfcases,
         ("ascend910b",), 4, (('pynative', 'kbk', 'ge'), (np.float32,))],
        [test_ops_swiglu.test_ops_swiglu_normal,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (0, 2, -1))],
        [test_ops_transpose.test_ops_transpose_normal,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_unbind.test_ops_unbind_forward,
         ("ascend910b",), 3, (('pynative',),)],
        [test_ops_unbind.test_ops_unbind_backward,
         ("ascend910b",), 8, (('pynative',),)],
        [test_ops_unfold.test_unfold,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_upsample_nearest.test_upsample_nearest,
         ("ascend910b",), 4, (("GRAPH_MODE_O0", "PYNATIVE_MODE"),)],
        [test_ops_var.test_ops_var_normal,
         ("ascend910b",), 4, (('pynative', 'KBK'),)],
        [test_diff.test_ops_diff_binary_cases,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_float_power.test_float_power_tensor_tensor_forward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (np.float32, np.float64))],
        [test_float_power.test_float_power_tensor_tensor_backward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (np.float32,))],
        [test_float_power.test_float_power_tensor_scalar_forward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (np.float32, np.float64))],
        [test_float_power.test_float_power_tensor_scalar_backward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (np.float32,))],
        [test_float_power.test_float_power_scalar_tensor_forward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (np.float32, np.float64))],
        [test_float_power.test_float_power_scalar_tensor_backward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE), (np.float32,))],
        [test_multinomial.test_multinomial_std,
         ("ascend910b",), 4, (('pynative', 'KBK',),)],
        [test_mint_nan_to_num.test_nan_to_num_std,
         ("ascend910b",), 32, (('pynative', 'KBK'),)],
        [test_nn_linear.test_mint_nn_linear_binary_cases_910b,
         ("ascend910b",), 21, (("pynative", "KBK"),)],
        [test_nn_AvgPool3d.test_mint_avg_pool3d_binary_cases,
         ("ascend910b",), 20, (("pynative", "KBK"),)],
        [test_addmv.test_mint_addmv_normal,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE,),)],
        [test_cdist.test_mint_cdist_binary_cases,
         ("ascend910b",), 4, (("KBK", "GRAPH"),)],
        [test_matmul.test_matmul_binary_cases,
         ("ascend910b",), 282, (("pynative", "KBK"),)],
        [test_nn_kldivloss.test_mint_nn_kldivloss_normal,
         ("ascend910b",), 4, (('pynative', 'KBK'),)],
    ],
    'binary_op': [
        op_db_group_cases_register(test_binary_ops, test_binary_ops.test_binary_op_reference_forward),
        op_db_group_cases_register(test_binary_ops, test_binary_ops.test_binary_op_reference_backward),
        op_db_group_cases_register(test_binary_ops, test_binary_ops.test_binary_op_reference_forward_kbk),
        op_db_group_cases_register(test_binary_ops, test_binary_ops.test_binary_op_reference_backward_kbk),
        op_db_group_cases_register(test_binary_ops, test_binary_ops.test_binary_op_type_promotion),
    ],
    'unary_op': [
        op_db_group_cases_register(test_unary_ops, test_unary_ops.test_unary_op_reference_forward),
        op_db_group_cases_register(test_unary_ops, test_unary_ops.test_unary_op_reference_backward),
        op_db_group_cases_register(test_unary_ops, test_unary_ops.test_unary_op_reference_forward_kbk),
        op_db_group_cases_register(test_unary_ops, test_unary_ops.test_unary_op_reference_backward_kbk),
    ],
    'reduction_op': [
        op_db_group_cases_register(test_reduction_ops, test_reduction_ops.test_reduction_op_reference_forward),
        op_db_group_cases_register(test_reduction_ops, test_reduction_ops.test_reduction_op_reference_backward),
        op_db_group_cases_register(test_reduction_ops, test_reduction_ops.test_reduction_op_reference_forward_kbk),
        op_db_group_cases_register(test_reduction_ops, test_reduction_ops.test_reduction_op_reference_backward_kbk),
    ],
    'other_op': [
        op_db_group_cases_register(test_other_ops, test_other_ops.test_other_op_reference_forward),
        op_db_group_cases_register(test_other_ops, test_other_ops.test_other_op_reference_backward),
        op_db_group_cases_register(test_other_ops, test_other_ops.test_other_op_reference_forward_kbk),
        op_db_group_cases_register(test_other_ops, test_other_ops.test_other_op_reference_backward_kbk),
    ]
}

ops_group_cases_registry_level1 = {
    # '''
    # cases in this registry will be running as level0 group cases on the gate
    # args:
    #     case_func(python function): case function to run
    #     platform(tuple(str)): use ascend or ascend910b
    #     memory(int): memory will be used during case running
    #     parameter(tuple(tuple(any))): a parameter list for cases, support muti-parameters
    # '''
    'ops': [
        [test_dynamic_quant.test_dynamic_quant_f16,
         ("ascend910b",), 16, (("KBK"),)],
        [test_dynamic_quant.test_dynamic_quant_bf16,
         ("ascend910b",), 16, (("KBK"),)],
        [test_func_conv2d.test_ops_conv2d_default,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_func_conv2d.test_conv2d_with_bf16,
         ("ascend910b",), 4, None],
        [test_func_conv2d.test_conv2d_backward,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_func_conv2d.test_conv2d_backward_fp16,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_func_conv2d.test_conv2d_vmap,
         ("ascend910b",), 4, ((ms.GRAPH_MODE, ms.PYNATIVE_MODE),)],
        [test_ops_concat.test_concat_binary_cases,
         ("ascend910b",), 1172, (('pynative', 'kbk'),)],
        [test_ops_gelu.test_ops_gelu_binary_cases,
         ("ascend910b",), 968, (('pynative', 'kbk', 'ge'),)],
        [test_nn_kldivloss.test_mint_nn_kldivloss_broadcast,
         ("ascend910b",), 4, (('pynative', 'KBK'),)],
    ],
    'binary_op': [
        op_db_group_cases_register(test_binary_ops, test_binary_ops.test_binary_op_dynamic_forward),
        op_db_group_cases_register(test_binary_ops, test_binary_ops.test_binary_op_dynamic_backward),
        op_db_group_cases_register(test_binary_ops, test_binary_ops.test_binary_op_error),
    ],
    'unary_op': [
        op_db_group_cases_register(test_unary_ops, test_unary_ops.test_unary_op_dynamic_forward),
        op_db_group_cases_register(test_unary_ops, test_unary_ops.test_unary_op_dynamic_backward),
    ],
    'reduction_op': [
        op_db_group_cases_register(test_reduction_ops, test_reduction_ops.test_reduction_op_dynamic_forward),
        op_db_group_cases_register(test_reduction_ops, test_reduction_ops.test_reduction_op_dynamic_backward),
    ],
    'other_op': [
        op_db_group_cases_register(test_other_ops, test_other_ops.test_other_op_dynamic_forward),
        op_db_group_cases_register(test_other_ops, test_other_ops.test_other_op_dynamic_backward),
    ]
}

def ops_group_cases_process(platform, level, group_name, *, max_task_num_per_process=None):
    '''
    group cases main process
    args:
        platform(str[ascend, ascend910b]): platform of mindspore
        level(str[level0, level1]): case level of gate
        group_name(str): group_name of cases on the gate
    keyword args:
        max_task_num_per_process(int, optional): max task number of each process, default: None.
    '''
    def run_group_cases(group_id, group_cases, memory_used, memory_threshold):
        msg = f"\nops group_cases_{group_id} with {len(group_cases)} cases start to running, all cases are below:\n"
        for case in group_cases:
            msg += f"case: {case}\n"
        msg += f"ops group_cases_{group_id} total running memory: {memory_used}M, memory threshold: {memory_threshold}M"
        print(msg)

        all_result = True
        result = run_cases_multi_process(group_cases)
        msg = f"group_cases_{group_id} have all been run, results of sub cases are below:\n"
        for ret in result:
            if ret[1]:
                msg += f"case: {ret[0]} pass.\n"
            else:
                msg += f"case: {ret[0]} fail.\n"
                all_result = False
        print(msg)
        assert all_result

    def get_param_list(rest_params, tmp_params, params_list):
        if rest_params == ():
            params_list.append(tmp_params)
            return
        params = rest_params[0:1][0]
        for param in params:
            new_params = copy.deepcopy(tmp_params)
            new_params.append(param)
            get_param_list(rest_params[1:], new_params, params_list)

    ops_group_cases_registry = {
        "level0": ops_group_cases_registry_level0,
        "level1": ops_group_cases_registry_level1,
    }

    memory_threshold_registry = {
        "ascend": 25600,
        "ascend910b": 51200,
        "cpu": 51200,
        "gpu": 19600,
    }

    if platform == "ascend":
        print("ascend don't support group cases yet.")
        return

    memory_used = 0
    group_id = 0
    group_cases = []
    if max_task_num_per_process:
        max_group_cases = get_max_worker() * max_task_num_per_process
    else:
        max_group_cases = np.iinfo(np.int32).max
    memory_threshold = memory_threshold_registry.get(platform)
    print(f"\nplatform: {platform}, max workers: {max_group_cases}, memory threshold: {memory_threshold}M"
          f"\nlevel: {level}, group_name: {group_name}\nStart group testing...")

    ops_group_cases = ops_group_cases_registry[level][group_name]

    for idx, case in enumerate(ops_group_cases):
        case_func = case[0]
        case_plat = case[1]
        case_mem = case[2]
        case_para = case[3]

        if platform not in case_plat:
            continue
        if case_para is not None:
            tmp_params = []
            params_list = []
            get_param_list(case_para, tmp_params, params_list)
            for param in params_list:
                if memory_used + case_mem > memory_threshold or len(group_cases) == max_group_cases:
                    run_group_cases(group_id, group_cases, memory_used, memory_threshold)
                    group_id += 1
                    group_cases = []
                    memory_used = 0
                sub_case = (case_func, *param)
                group_cases.append(sub_case)
                memory_used += case_mem
        else:
            if memory_used + case_mem > memory_threshold or len(group_cases) == max_group_cases:
                run_group_cases(group_id, group_cases, memory_used, memory_threshold)
                group_id += 1
                group_cases = []
                memory_used = 0
            sub_case = (case_func,)
            group_cases.append(sub_case)
            memory_used += case_mem
        if  idx == len(ops_group_cases) - 1:
            run_group_cases(group_id, group_cases, memory_used, memory_threshold)


def get_local_platform():
    device = ms.context.get_context('device_target').lower()
    if device == 'ascend':
        soc_name = MSContext.get_instance().get_ascend_soc_version()
        device_name = 'ascend' if soc_name != 'ascend910b' else soc_name
    elif device == 'cpu':
        device_name = 'cpu'
    elif device == 'gpu':
        device_name = 'gpu'
    else:
        raise ValueError(f"Invalid device: {device}, expected: 'ascend', 'cpu', 'gpu'.")
    return device_name


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_ops_group_case_ascend910b_level0():
    """
    Feature: Test all level0 cases on ascend910b.
    Description: Run ascend910b level0 cases on the gate.
    Expectation: success
    """
    ops_group_cases_process("ascend910b", "level0", "ops", max_task_num_per_process=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_ops_group_case_ascend910b_level1():
    """
    Feature: Test all level1 cases on ascend910b.
    Description: Run ascend910b level0 cases on the gate.
    Expectation: success
    """
    ops_group_cases_process("ascend910b", "level1", "ops", max_task_num_per_process=1)


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_binary_op_group_case_level0():
    """
    Feature: Test all level0 binary_op cases.
    Description: Run level0 binary_op cases on the gate.
    Expectation: success
    """
    platform = get_local_platform()
    ops_group_cases_process(platform, "level0", "binary_op")


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_binary_op_group_case_level1():
    """
    Feature: Test all level1 binary_op cases.
    Description: Run level1 binary_op cases on the gate.
    Expectation: success
    """
    platform = get_local_platform()
    ops_group_cases_process(platform, "level1", "binary_op")


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_unary_op_group_case_level0():
    """
    Feature: Test all level0 unary_op cases.
    Description: Run level0 unary_op cases on the gate.
    Expectation: success
    """
    platform = get_local_platform()
    ops_group_cases_process(platform, "level0", "unary_op")


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_reduction_op_group_case_level0():
    """
    Feature: Test all level0 reduction_op cases.
    Description: Run level0 reduction_op cases on the gate.
    Expectation: success
    """
    platform = get_local_platform()
    ops_group_cases_process(platform, "level0", "reduction_op")


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_reduction_op_group_case_level1():
    """
    Feature: Test all level1 reduction_op cases.
    Description: Run level1 reduction_op cases on the gate.
    Expectation: success
    """
    platform = get_local_platform()
    ops_group_cases_process(platform, "level1", "reduction_op")


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_other_op_group_case_level0():
    """
    Feature: Test all level0 other_op cases.
    Description: Run level0 other_op cases on the gate.
    Expectation: success
    """
    platform = get_local_platform()
    ops_group_cases_process(platform, "level0", "other_op")


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.no_taskset
def test_other_op_group_case_level1():
    """
    Feature: Test all level1 other_op cases.
    Description: Run level1 other_op cases on the gate.
    Expectation: success
    """
    platform = get_local_platform()
    ops_group_cases_process(platform, "level1", "other_op")
