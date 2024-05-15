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
Parse Update Weights Name.
"""
import re
import os
def _maybe_map_sgm_blocks_to_diffusers(name, layers_per_block=2, delimiter="_", block_slice_pos=5):
    '''
    convert name like input_blocks.1.1_xxx to input_blocks.1.resnets_xxx
    '''
    # 1. get all state_dict_keys
    sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]
    inner_block_map = ["resnets", "attentions", "upsamplers"]

    if not any([pattern in name for pattern in sgm_patterns]):
        return name

    layer_id = int(name.split(delimiter)[:block_slice_pos][-1])

    # Rename keys accordingly
    if sgm_patterns[0] in name: # 0:input_blocks
        block_id = (layer_id - 1) // (layers_per_block + 1)
        layer_in_block_id = (layer_id - 1) % (layers_per_block + 1)

        inner_block_id = int(name.split(delimiter)[block_slice_pos])
        inner_block_key = inner_block_map[inner_block_id] if "op" not in name else "downsamplers"
        inner_layers_in_block = str(layer_in_block_id) if "op" not in name else "0"
        new_name = delimiter.join(
            name.split(delimiter)[: block_slice_pos - 1]
            + [str(block_id), inner_block_key, inner_layers_in_block]
            + name.split(delimiter)[block_slice_pos + 1 :]
        )
        return new_name

    if sgm_patterns[1] in name: # 1:middle_block
        key_part = None
        if layer_id == 0:
            key_part = [inner_block_map[0], "0"]
        elif layer_id == 1:
            key_part = [inner_block_map[1], "0"]
        elif layer_id == 2:
            key_part = [inner_block_map[0], "1"]
        else:
            raise ValueError(f"Invalid middle block id {layer_id}.")

        new_name = delimiter.join(
            name.split(delimiter)[: block_slice_pos - 1] + key_part + name.split(delimiter)[block_slice_pos:]
        )
        return new_name

    if sgm_patterns[2] in name: # 2:output_blocks
        block_id = layer_id // (layers_per_block + 1)
        layer_in_block_id = layer_id % (layers_per_block + 1)
        name_splites = name.split(delimiter)
        if len(name_splites) <= block_slice_pos:
            raise ValueError("Invalid name")

        inner_block_id = int(name_splites[block_slice_pos])
        inner_block_key = inner_block_map[inner_block_id]
        inner_layers_in_block = str(layer_in_block_id) if inner_block_id < 2 else "0"
        new_name = delimiter.join(
            name.split(delimiter)[: block_slice_pos - 1]
            + [str(block_id), inner_block_key, inner_layers_in_block]
            + name.split(delimiter)[block_slice_pos + 1 :]
        )
        return new_name

    return name

def _judge_name_begin(name):
    return not name.startswith("lora_unet_") and not name.startswith("lora_te1_") \
        and not name.startswith("lora_te2_") and not name.startswith("lora_te_")

def _convert_kohya_name(name):
    '''
    convert name like input_blocks_xxxx to down_blocks_xxxx
    '''
    diffusers_name = name
    lora_name = name.split(".")[0]

    if _judge_name_begin(lora_name):
        return diffusers_name

    diffusers_name = name.replace("lora_te1_", "")
    diffusers_name = diffusers_name.replace("lora_te2_", "")
    diffusers_name = diffusers_name.replace("lora_te_", "")
    diffusers_name = diffusers_name.replace("lora_unet_", "").replace("_", ".")
    diffusers_name = diffusers_name.replace("text.model", "text_model")

    if "input.blocks" in diffusers_name:
        diffusers_name = diffusers_name.replace("input.blocks", "down_blocks")
    else:
        diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")

    if "middle.block" in diffusers_name:
        diffusers_name = diffusers_name.replace("middle.block", "mid_block")
    else:
        diffusers_name = diffusers_name.replace("mid.block", "mid_block")
    if "output.blocks" in diffusers_name:
        diffusers_name = diffusers_name.replace("output.blocks", "up_blocks")
    else:
        diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")

    diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
    diffusers_name = diffusers_name.replace("to.q", "to_q")
    diffusers_name = diffusers_name.replace("to.k", "to_k")
    diffusers_name = diffusers_name.replace("to.v", "to_v")
    diffusers_name = diffusers_name.replace("to.out.0", "to_out")
    diffusers_name = diffusers_name.replace("proj.in", "proj_in")
    diffusers_name = diffusers_name.replace("proj.out", "proj_out")
    diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    diffusers_name = diffusers_name.replace("q.proj", "q_proj")
    diffusers_name = diffusers_name.replace("k.proj", "k_proj")
    diffusers_name = diffusers_name.replace("v.proj", "v_proj")
    diffusers_name = diffusers_name.replace("out.proj", "out_proj")

    # SDXL specificity.
    if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name:
        pattern = r"\.\d+(?=\D*$)"
        diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
    if ".in." in diffusers_name:
        diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
    if ".out." in diffusers_name:
        diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
    if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
        diffusers_name = diffusers_name.replace("op", "conv")
    if "skip" in diffusers_name:
        diffusers_name = diffusers_name.replace("skip.connection", "conv_shortcut")

    # LyCORIS specificity.
    if "time.emb.proj" in diffusers_name:
        diffusers_name = diffusers_name.replace("time.emb.proj", "time_emb_proj")
    if "conv.shortcut" in diffusers_name:
        diffusers_name = diffusers_name.replace("conv.shortcut", "conv_shortcut")

    # General coverage.
    if "transformer_blocks" in diffusers_name:
        if "attn1" in diffusers_name or "attn2" in diffusers_name:
            diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
            diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
    return diffusers_name

def _rename_variable_weight(name):
    """Rename variable weight"""
    if not name.endswith("weight") and not name.endswith("alpha"):
        raise RuntimeError("variable is not norm name, now only support **weight")
    custom_prefix = None
    if name.startswith("model.diffusion"):
        name_parts = name.split('.')
        custom_prefix_parts = [name_parts[2] + '.' + name_parts[3], name_parts[2] +
                               '.' + name_parts[3] + '.' + name_parts[4]]
        custom_prefix = '/'.join(custom_prefix_parts) + '/'
        name = '.'.join(name_parts[5:])
        name = name.replace("lora_up.", '')
        name = name.replace("lora_down.", '')
        name = name.replace("net.0", "net.net.0")
    name = name.replace('unet.', '')
    name = _maybe_map_sgm_blocks_to_diffusers(name)

    name = _convert_kohya_name(name)
    name = name.replace("out_0", "out").replace("out.0", "out")
    name = name.replace(".down.", ".").replace(".up.", ".")

    name = name.replace('_lora', '')
    name = name.replace('lora.', '')
    name = name.replace('te1.', '')
    name = name.replace('te2.', '')
    name = name.replace('te.', '')
    name = name.replace('processor.', '')
    name = name.replace('text_encoder.', '')
    name = name.replace('text_encoder_2.', '')
    name = name.replace('lora_linear_layer.', '')
    name = name.replace('linear_layer.', '')
    name_split = name.split('.')
    name_split.pop()
    merged_name = []
    index = len(name_split) - 1
    while index >= 0:
        if name_split[index].isdigit():
            merged_name.append(name_split[index-1] + '.' + name_split[index])
            index -= 2
        else:
            merged_name.append(name_split[index])
            index -= 1

    merged_name.reverse()
    new_name = '/'.join(merged_name)
    new_name = new_name.replace('to_out', 'to_out.0')
    new_name = new_name.replace('to_out.0', 'to_out/to_out.0') if custom_prefix is not None else new_name
    return "/" + new_name if custom_prefix is None else "/" + custom_prefix + new_name

def _get_variable_weights_name(name_list_file):
    """Get variable weights name"""
    if not os.path.exists(name_list_file):
        raise RuntimeError("variable weight name list is not exists.")
    name_map = {}
    new_name_str = ""
    new_names = []
    with open(name_list_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[-1] == "\n":
                line = line[:-1]
            line_split = line.split(',')
            if len(line_split) == 2:
                name_map[line_split[0]] = line_split[1]
                new_name = line_split[1]
            elif len(line_split) == 1:
                new_name = _rename_variable_weight(line)
            else:
                raise RuntimeError("only support 1 or 2 row name list, current row num:",
                                   len(line_split), ' !')
            if new_name not in new_names:
                new_names.append(new_name)
                new_name_str += ',' + new_name
    return new_name_str[1:], name_map

def _parse_update_weight_config_name(name_list_file):
    """Parse update weight config name"""
    with open(name_list_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            if "variable_weights_file" in line:
                name_list_file = line.split('=')[1]
                return _get_variable_weights_name(name_list_file)
    return None
