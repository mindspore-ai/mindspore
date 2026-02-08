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
"""Run mcore GPTModel UT of inference with configurable parameters via args"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.communication import init, comm_func

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference import parallel_state as ps
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.models.utils import jit

from data_gen_utils import (
    get_init_params,
    DEFAULT_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
)

SCRIPT_DIR = Path(__file__).parent.resolve()


class GPTModelRunner:
    """Class to manage Transformer Block module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.is_prefill = self.args.is_prefill

        # Model dimensions
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length
        self.vocab_size = self.args.vocab_size
        self.max_position_embeddings = self.args.max_position_embeddings
        self.hidden_size = self.args.hidden_size
        self.ffn_hidden_size = self.args.ffn_hidden_size
        self.num_attention_heads = self.args.num_attention_heads
        self.num_layers = self.args.num_layers

        # get layer spec params
        self.num_experts = self.args.num_experts
        self.moe_grouped_gemm = self.args.moe_grouped_gemm
        self.qk_layernorm = self.args.qk_layernorm
        self.multi_latent_attention = self.args.multi_latent_attention
        self.qk_l2_norm = self.args.qk_l2_norm
        self.sandwich_norm = self.args.sandwich_norm

        # parallel config
        self.tensor_parallel = self.args.tensor_parallel
        self.pipeline_parallel = self.args.pipeline_parallel

        self.compute_dtype = mstype.bfloat16
        self.params_dtype = mstype.bfloat16

        init_params = get_init_params(self.is_prefill, self.batch_size, self.seq_length, self.vocab_size)

        self.input_ids = init_params.get("input_ids")
        self.positions = init_params.get("positions")
        self.attention_mask = init_params.get("attention_mask")
        if self.is_prefill:
            self.attention_mask = self.attention_mask.astype(self.compute_dtype)
        self.batch_valid_length = init_params.get("batch_valid_length")
        self.q_seq_lens = init_params.get("q_seq_lens")
        self.context_lens_tensor = init_params.get("context_lens_tensor")
        self.block_tables = init_params.get("block_tables")
        self.slot_mapping = init_params.get("slot_mapping")

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context
        self.model_comm_pgs = ModelCommProcessGroups.get_default_model_comm_pgs()
        if self.rank_id is not None:
            init()
            ps.initialize_model_parallel(tensor_model_parallel_size=self.tensor_parallel,
                                         pipeline_model_parallel_size=self.pipeline_parallel)
            self.model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups()

        # Transformer config
        self.config = TransformerConfig(
            tensor_model_parallel_size=self.tensor_parallel,
            pipeline_model_parallel_size=self.pipeline_parallel,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.num_attention_heads,
            add_bias_linear=False,
            gated_linear_unit=True,
            hidden_act="silu",
            normalization="RMSNorm",
            num_layers=self.num_layers,
            num_blocks=DEFAULT_NUM_BLOCKS,
            block_size=DEFAULT_BLOCK_SIZE,
            use_flash_attention=True,
            compute_dtype='bf16',
            params_dtype='bf16',
            pre_process=ps.is_pipeline_first_stage(),
            post_process=ps.is_pipeline_last_stage(),
        )

        self.model = self.build_model()
        if not self.is_prefill:
            self.model.add_flags(is_prefill=False)
            self.model.decoder.add_flags(is_prefill=False)
            for layer in self.model.decoder.layers:
                layer.self_attention.core_attention.add_flags(is_prefill=False)

    def build_model(self):
        """Build Transformer Layer module"""
        layer_spec = get_gpt_layer_local_spec(
            num_experts=self.num_experts,
            moe_grouped_gemm=self.moe_grouped_gemm,
            qk_layernorm=self.qk_layernorm,
            multi_latent_attention=self.multi_latent_attention,
            normalization=self.config.normalization,
            qk_l2_norm=self.qk_l2_norm,
            sandwich_norm=self.sandwich_norm,
        )
        net = GPTModel(config=self.config,
                       transformer_layer_spec=layer_spec,
                       vocab_size=self.vocab_size,
                       max_sequence_length=self.max_position_embeddings,
                       position_embedding_type=self.config.position_embedding_type,
                       pre_process=self.config.pre_process,
                       post_process=self.config.post_process,
                       model_comm_pgs=self.model_comm_pgs)
        return net

    def forward(self, input_ids, hidden_states=None, positions=None, batch_valid_length=None,
                context_lens_tensor=None, q_seq_lens=None, block_tables=None, slot_mapping=None,
                attention_mask=None, attn_metadata=None, key_cache=None, value_cache=None):
        """Forward pass for pipeline parallelism"""
        if not getattr(self, "jit_forward", None):
            self.jit_forward = jit(self.model)

        logits = ms.mint.zeros((self.batch_size, self.vocab_size), dtype=ms.float32)
        hidden_states_shape = (self.batch_size * self.seq_length,
                               self.hidden_size) if self.is_prefill else (self.batch_size, self.hidden_size)
        hidden_states = ms.mint.zeros(hidden_states_shape, dtype=self.compute_dtype)

        if not self.config.pre_process:
            comm_func.recv(hidden_states,
                           src=ps.get_pipeline_model_parallel_prev_rank(),
                           group=ps.get_pipeline_model_parallel_group().group)

        if not self.config.post_process:
            hidden_states = self.jit_forward(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                batch_valid_length=batch_valid_length,
                context_lens_tensor=context_lens_tensor,
                q_seq_lens=q_seq_lens,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                attention_mask=attention_mask,
                attn_metadata=attn_metadata,
                key_cache=key_cache,
                value_cache=value_cache
            )
            comm_func.send(hidden_states,
                           dst=ps.get_pipeline_model_parallel_next_rank(),
                           group=ps.get_pipeline_model_parallel_group().group)

        if self.config.post_process:
            logits = self.jit_forward(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                batch_valid_length=batch_valid_length,
                context_lens_tensor=context_lens_tensor,
                q_seq_lens=q_seq_lens,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                attention_mask=attention_mask,
                attn_metadata=attn_metadata,
                key_cache=key_cache,
                value_cache=value_cache
            )
        if ps.get_pipeline_model_parallel_world_size() > 1:
            comm_func.all_reduce(logits, group=ps.get_pipeline_model_parallel_group().group)
        return logits

    def run(self):
        """Run the model with given inputs"""
        self.net = self.build_model()

        kv_cache_shape = (
            DEFAULT_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            self.num_attention_heads // self.args.tensor_parallel,
            self.hidden_size // self.num_attention_heads)
        key_cache = []
        value_cache = []
        for _ in range(self.config.num_layers):
            key_cache.append(ms.mint.zeros(kv_cache_shape, dtype=self.compute_dtype))
            value_cache.append(ms.mint.zeros(kv_cache_shape, dtype=self.compute_dtype))

        output = self.forward(input_ids=self.input_ids,
                              positions=self.positions,
                              attention_mask=self.attention_mask,
                              batch_valid_length=self.batch_valid_length,
                              context_lens_tensor=self.context_lens_tensor,
                              q_seq_lens=self.q_seq_lens,
                              block_tables=self.block_tables,
                              slot_mapping=self.slot_mapping,
                              key_cache=ms.mutable(key_cache),
                              value_cache=ms.mutable(value_cache)
                              )
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.asnumpy().astype(np.float16) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run GPTModel test")
    parser.add_argument("--is_prefill", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--max_position_embeddings", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--ffn_hidden_size", type=int, default=64)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--moe_grouped_gemm", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--qk_layernorm", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--multi_latent_attention", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--qk_l2_norm", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--sandwich_norm", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--pipeline_parallel", type=int, default=1)

    args = parser.parse_args()

    # Prepare environment
    ms.set_deterministic(True)
    ms.set_device("Ascend")
    ms.set_seed(124)
    os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = "PagedAttention"
    os.environ["RUN_MODE"] = "predict"

    # Prepare input
    runner = GPTModelRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
