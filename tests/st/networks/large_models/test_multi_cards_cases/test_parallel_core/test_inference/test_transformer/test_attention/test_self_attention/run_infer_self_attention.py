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
"""Test Attention"""
import argparse
import os
import sys

import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, mint, Parameter, ops
from mindspore.communication import init, get_rank

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    QKVParallelLinear,
    RowParallelLinear,
)
from mindformers.parallel_core.inference.transformer.attention import (
    SelfAttention,
    SelfAttentionSubmodules,
)
from mindformers.parallel_core.inference.transformer.dot_product_attention import DotProductAttention
from mindformers.parallel_core.inference.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups
from data_gen_utils import (
    get_init_params,
    BLOCK_SIZE, NUM_BLOCKS)


class SelfAttnRunner:
    """Class to manage Self_Attn module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.use_flash_attention = self.args.use_flash_attention
        self.batch_size = self.args.batch_size
        self.n_kv_heads = self.args.num_query_groups
        self.num_heads = self.args.num_heads

        self.prefill_seq_length = self.args.prefill_seq_len
        self.decoder_seq_length = self.args.decode_seq_len

        self.hidden_size = self.args.hidden_size
        self.head_dim = int(self.hidden_size / self.args.num_heads)
        self.is_prefill = True
        self.tensor_parallel = self.args.tensor_parallel
        self.kv_cache_shape = (
            NUM_BLOCKS, BLOCK_SIZE, self.n_kv_heads, self.head_dim
        )

        init_params = get_init_params(n_kv_heads=self.args.num_query_groups)
        self.prefill_hidden_states = ms.Tensor(init_params.get("prefill_hidden_states"),
                                               dtype=mstype.bfloat16)
        prefill_shape = (-1, self.num_heads * (self.hidden_size // self.num_heads))
        self.prefill_hidden_states = self.prefill_hidden_states.reshape(prefill_shape)
        self.decoder_hidden_states = ms.Tensor(init_params.get("decoder_hidden_states"),
                                               dtype=mstype.bfloat16)
        decode_shape = (-1, self.num_heads * (self.hidden_size // self.num_heads))
        self.decoder_hidden_states = self.decoder_hidden_states.reshape(decode_shape)
        self.prefill_slot_mapping = Tensor(init_params.get("prefill_slot_mapping"), dtype=mstype.int32)
        self.decoder_slot_mapping = Tensor(init_params.get("decoder_slot_mapping"), dtype=mstype.int32)

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.qkv_weight = init_params.get("qkv_weight")
        self.proj_weight = init_params.get("proj_weight")

        self.block_tables = Tensor(init_params.get("block_tables"), mstype.int32)
        self.generate_mask()

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context
        self.model_comm_pgs = ModelCommProcessGroups.get_default_model_comm_pgs()
        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=self.args.tensor_parallel)
            self.model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups(required_groups=['tp'])

        self.config = self.get_self_attention_config()

    def generate_mask(self):
        """ generate flash_attn_mask and batch_valid_length for prefill and decode"""
        if self.is_prefill:
            self.flash_attn_mask = mint.ones((self.prefill_seq_length,
                                              self.prefill_seq_length), dtype=mstype.float16)
            self.flash_attn_mask = (mint.triu(self.flash_attn_mask, diagonal=1)
                                    .astype(mstype.bfloat16))
            self.batch_valid_length = Tensor(np.ones((self.batch_size,)) * self.prefill_seq_length, dtype=mstype.int32)
        else:
            self.flash_attn_mask = mint.ones((self.decoder_seq_length,
                                              self.decoder_seq_length), dtype=mstype.float16)
            self.flash_attn_mask = mint.triu(self.flash_attn_mask, diagonal=1).astype(mstype.bfloat16)
            self.batch_valid_length = Tensor(
                np.ones((self.batch_size,)) * (self.prefill_seq_length + self.decoder_seq_length),
                dtype=mstype.int32)

    def get_self_attention_config(self):
        """Generate config for SelfAttention test."""
        config = TransformerConfig(
            num_attention_heads=self.num_heads,
            num_query_groups=self.args.num_query_groups,
            hidden_size=self.hidden_size,
            sequence_parallel=False,
            num_layers=1,
            seq_length=self.prefill_seq_length,
            compute_dtype="bfloat16",
            softmax_compute_dtype="bfloat16",
            layernorm_compute_dtype="bfloat16",
            add_qkv_bias=False,
            add_bias_linear=False,
            use_flash_attention=self.use_flash_attention,
            num_blocks=NUM_BLOCKS,
            block_size=BLOCK_SIZE,
            tensor_model_parallel_size=self.tensor_parallel
        )

        return config

    @staticmethod
    def _get_self_attn_spec(use_flash_attention):
        """Construct test self_attn spec."""

        if use_flash_attention:
            self_attn = ModuleSpec(module=SelfAttention,
                                   submodules=SelfAttentionSubmodules(
                                       core_attention=FlashAttention,
                                       linear_proj=RowParallelLinear,
                                       linear_qkv=QKVParallelLinear,
                                       q_layernorm=None,
                                       k_layernorm=None))
        else:
            self_attn = ModuleSpec(module=SelfAttention,
                                   submodules=SelfAttentionSubmodules(
                                       core_attention=DotProductAttention,
                                       linear_proj=RowParallelLinear,
                                       linear_qkv=QKVParallelLinear,
                                       q_layernorm=None,
                                       k_layernorm=None))
        return self_attn

    def build_model(self):
        """Build Self_Attention"""
        net = build_module(
            self._get_self_attn_spec(self.use_flash_attention),
            config=self.config,
            attn_mask_type=None,
            layer_number=1,
            model_comm_pgs=self.model_comm_pgs,
        )
        self._load_weight(net)
        return net

    def _load_weight(self, net):
        """load weights for self_attention"""
        rank_id = get_rank()
        into_weight = {}

        def split(weight, split_axis=0):
            split_size = weight.shape[split_axis] // self.tensor_parallel
            start = rank_id * split_size
            stop = (rank_id + 1) * split_size
            return weight[start:stop] if split_axis == 0 else weight[:, start:stop]

        w_q = self.qkv_weight[:self.head_dim * self.num_heads, :]
        w_k = self.qkv_weight[self.head_dim * self.num_heads:
                              self.head_dim * (self.num_heads + self.n_kv_heads), :]
        w_v = self.qkv_weight[self.head_dim * (self.num_heads + self.n_kv_heads):
                              self.head_dim * (self.num_heads + 2 * self.n_kv_heads), :]
        w_q_shard = split(w_q)
        w_k_shard = split(w_k)
        w_v_shard = split(w_v)
        w_o = split(self.proj_weight, split_axis=1)
        w_qkv_shard = np.concatenate([w_q_shard, w_k_shard, w_v_shard], axis=0)

        into_weight["linear_qkv.weight"] = Parameter(w_qkv_shard)
        into_weight["linear_proj.weight"] = Parameter(w_o)

        ms.load_param_into_net(net, into_weight)

    def run(self):
        """Run self_attn with given inputs"""

        #prefill
        net = self.build_model()

        kv_cache_shape = (
            NUM_BLOCKS,
            BLOCK_SIZE,
            self.config.num_query_groups // self.tensor_parallel,
            self.config.hidden_size // self.config.num_attention_heads)
        key_cache = mint.zeros(kv_cache_shape, dtype=self.config.compute_dtype)
        value_cache = mint.zeros(kv_cache_shape, dtype=self.config.compute_dtype)

        prefill_output = net(self.prefill_hidden_states,
                             attention_mask=self.flash_attn_mask,
                             slot_mapping=self.prefill_slot_mapping,
                             actual_seq_qlen=self.batch_valid_length,
                             actual_seq_kvlen=self.batch_valid_length,
                             key_cache=key_cache,
                             value_cache=value_cache
                             )

        #decode
        self.is_prefill = False
        self.generate_mask()
        net.phase = "increment"
        net.add_flags(is_prefill=False)
        if self.use_flash_attention:
            net.core_attention.add_flags(is_prefill=False)

        decode_output = net(self.decoder_hidden_states,
                            attention_mask=None,
                            block_tables=self.block_tables,
                            slot_mapping=self.decoder_slot_mapping,
                            batch_valid_length=self.batch_valid_length,
                            context_lens_tensor=self.batch_valid_length,
                            key_cache=key_cache,
                            value_cache=value_cache
                            )

        output_ms = {"prefill_output": prefill_output, "decode_output": decode_output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {
                k: v.asnumpy().astype(np.float16)
                for k, v in output_ms.items() if v is not None
            }
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument("--prefill_seq_len", type=int, default=2)
    parser.add_argument("--decode_seq_len", type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_query_groups', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--use_flash_attention', type=lambda x: x.lower() == "true", default=True)
    parser.add_argument('--output_path', type=str, default="output_ms.npz")
    parser.add_argument('--tensor_parallel', type=int, default=1)

    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    jit_level = "O0"
    infer_boost = "on"
    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    seed_value = 2025
    ms.set_seed(seed_value)
    np.random.seed(seed_value)

    runner = SelfAttnRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
