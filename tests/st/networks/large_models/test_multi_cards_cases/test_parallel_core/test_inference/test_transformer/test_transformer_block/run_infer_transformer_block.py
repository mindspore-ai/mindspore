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
"""Run mcore Transformer Block UT of inference with configurable parameters via args"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.communication import init

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups
from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

from data_gen_utils import (
    get_init_params,
    DEFAULT_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
)

SCRIPT_DIR = Path(__file__).parent.resolve()


class TransformerBlockRunner:
    """Class to manage Transformer Block module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser

        # Model dimensions
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length
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

        self.compute_dtype = mstype.bfloat16
        self.params_dtype = mstype.bfloat16

        init_params = get_init_params(self.batch_size, self.seq_length, self.hidden_size)

        self.hidden_states = ms.Tensor(init_params.get("hidden_states"), dtype=mstype.bfloat16)
        self.attention_mask = ms.Tensor(
            np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * -10000.0, dtype=self.compute_dtype)
        self.batch_valid_length = ms.Tensor(np.ones((self.batch_size,)).astype(np.int32))
        self.q_seq_lens = ms.Tensor(np.ones((self.batch_size,)).astype(np.int32))
        self.context_lens_tensor = ms.Tensor(np.zeros((self.batch_size,)).astype(np.int32))
        self.block_tables = ms.Tensor(np.ones((self.batch_size, DEFAULT_NUM_BLOCKS)).astype(np.int32))
        self.slot_mapping = ms.Tensor(np.ones((self.batch_size * self.seq_length,)).astype(np.int32))

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

        # Transformer config
        self.config = TransformerConfig(
            tensor_model_parallel_size=self.args.tensor_parallel,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
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
            params_dtype='bf16'
        )

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
        net = TransformerBlock(self.config, layer_spec,
                               model_comm_pgs=self.model_comm_pgs,)
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

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

        output = net(hidden_states=self.hidden_states,
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
    parser = argparse.ArgumentParser(description="Run Transformer Block test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=2)
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

    # Prepare input
    runner = TransformerBlockRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
