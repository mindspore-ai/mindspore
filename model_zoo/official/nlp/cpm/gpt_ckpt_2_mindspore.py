# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Checkpoint."""
import numpy as np
import torch

from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

param_names = {
    "word_embeddings.weight": "word_embedding.embedding_table",
    "position_embeddings.weight": "position_embedding.position_embedding_table",
    "transformer.final_layernorm.weight": "transformer.final_layernorm.gamma",
    "transformer.final_layernorm.bias": "transformer.final_layernorm.beta",
}
for i in range(0, 32):
    param_names["transformer.layers." + str(i) + ".attention.query_key_value.weight"] = ""
    param_names["transformer.layers." + str(i) + ".attention.query_key_value.bias"] = ""
    param_names["transformer.layers." + str(i) + ".attention.dense.weight"] = "transformer.layers." + str(
        i) + ".masked_multi_head_attention.masked_self_attention.dense.weight"
    param_names["transformer.layers." + str(i) + ".attention.dense.bias"] = "transformer.layers." + str(
        i) + ".masked_multi_head_attention.masked_self_attention.dense.bias"
    param_names["transformer.layers." + str(i) + ".input_layernorm.weight"] = "transformer.layers." + str(
        i) + ".masked_multi_head_attention.layernorm.gamma"
    param_names["transformer.layers." + str(i) + ".input_layernorm.bias"] = "transformer.layers." + str(
        i) + ".masked_multi_head_attention.layernorm.beta"
    param_names["transformer.layers." + str(i) + ".mlp.dense_h_to_4h.weight"] = "transformer.layers." + str(
        i) + ".mlp.dense_fc.weight"
    param_names["transformer.layers." + str(i) + ".mlp.dense_h_to_4h.bias"] = "transformer.layers." + str(
        i) + ".mlp.dense_fc.bias"
    param_names["transformer.layers." + str(i) + ".mlp.dense_4h_to_h.weight"] = "transformer.layers." + str(
        i) + ".mlp.dense_proj.weight"
    param_names["transformer.layers." + str(i) + ".mlp.dense_4h_to_h.bias"] = "transformer.layers." + str(
        i) + ".mlp.dense_proj.bias"
    param_names["transformer.layers." + str(i) + ".post_attention_layernorm.weight"] = "transformer.layers." + str(
        i) + ".mlp.layernorm.gamma"
    param_names["transformer.layers." + str(i) + ".post_attention_layernorm.bias"] = "transformer.layers." + str(
        i) + ".mlp.layernorm.beta"


def torch2ms(torch_ckpt):
    """Translate the model to mindspore checkpoint."""
    torch_param_dict = torch.load(torch_ckpt, map_location=torch.device('cpu'))['module']

    with open("weight_torch.txt", "w") as f:
        for key, value in torch_param_dict.items():
            print(f'torch key   = {key}')
            f.write(key + ' ' + 'dtype=' + str(value.dtype) + "\n")
            print(f'value = {value}')
    print("-----------------------------------------")
    new_params_list = []
    for torch_name in torch_param_dict:
        ms_param_dict = {}

        torch_value = torch_param_dict[torch_name]
        ms_name = param_names[torch_name]

        if "word_embeddings.weight" in torch_name:
            ms_param_dict['name'] = ms_name
            ms_param_dict['data'] = Tensor(torch_value.numpy().astype(np.float32))
            new_params_list.append(ms_param_dict)
            print(f'torch_name = {torch_name}, ms_name = {ms_name}, fp32')
        elif "layernorm" in torch_name:
            ms_param_dict['name'] = ms_name
            ms_param_dict['data'] = Tensor(torch_value.numpy().astype(np.float32))
            new_params_list.append(ms_param_dict)
            print(f'torch_name = {torch_name}, ms_name = {ms_name}, fp32')
        elif "query_key_value" not in torch_name:
            ms_param_dict['name'] = ms_name
            ms_param_dict['data'] = Tensor(torch_value.numpy().astype(np.float32))
            new_params_list.append(ms_param_dict)
            print(f'torch_name = {torch_name}, ms_name = {ms_name}, fp16')
        else:
            _, _, index, _, _, end = torch_name.split(".")

            prefix = "transformer.layers."
            q_mid = ".masked_multi_head_attention.masked_self_attention.dense1."
            k_mid = ".masked_multi_head_attention.masked_self_attention.dense2."
            v_mid = ".masked_multi_head_attention.masked_self_attention.dense3."

            q_name = prefix + str(index) + q_mid + end
            k_name = prefix + str(index) + k_mid + end
            v_name = prefix + str(index) + v_mid + end
            print(f'q_name = {q_name}')
            print(f'k_name = {k_name}')
            print(f'v_name = {v_name}')

            query, key, value = torch_param_dict[torch_name].chunk(3, dim=0)
            print(f"query shape = {query.shape}, key shape = {key.shape}, value shape = {value.shape}")
            q_param_dict = {}
            k_param_dict = {}
            v_param_dict = {}
            q_param_dict['name'] = q_name
            k_param_dict['name'] = k_name
            v_param_dict['name'] = v_name
            q_param_dict['data'] = Tensor(query.numpy().astype(np.float32))
            k_param_dict['data'] = Tensor(key.numpy().astype(np.float32))
            v_param_dict['data'] = Tensor(value.numpy().astype(np.float32))
            new_params_list.append(q_param_dict)
            new_params_list.append(k_param_dict)
            new_params_list.append(v_param_dict)
            print(f'torch_name = {torch_name}, ms_name = {ms_name}, fp16')

    save_checkpoint(new_params_list, '/home/cpm_mindspore_1p_fp32.ckpt')


if __name__ == '__main__':
    original_ckpt = "/home/CPM-large_MP1/iter_0080000/mp_rank_00_model_states.pt"

    torch2ms(original_ckpt)
