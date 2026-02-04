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
"""Run mcore MLP UT of inference with configurable parameters via args"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

import mindspore as ms
from mindspore import Parameter
import mindspore.common.dtype as mstype
from mindspore.communication import init, get_rank

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import build_module
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_mlp_module_spec
from data_gen_utils import (
    get_init_params,
    INPUT_SIZE,
)

SCRIPT_DIR = Path(__file__).parent.resolve()


class MLPRunner:
    """Class to manage MLP module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.has_bias = self.args.has_bias
        self.gated_linear_unit = self.args.gated_linear_unit

        self.input_size = self.args.input_size
        self.ffn_hidden_size = self.args.ffn_hidden_size
        self.compute_dtype = mstype.bfloat16
        self.params_dtype = mstype.float32

        init_params = get_init_params(INPUT_SIZE, self.args.ffn_hidden_size)

        self.input = ms.Tensor(init_params.get("input"), dtype=mstype.bfloat16)
        if self.gated_linear_unit:
            self.fc1_weight = init_params.get("fc1_gate_weight")
        else:
            self.fc1_weight = init_params.get("fc1_no_gate_weight")
        if self.has_bias:
            self.fc2_bias = init_params.get("fc2_bias")
            if self.gated_linear_unit:
                self.fc1_bias = init_params.get("fc1_gate_bias")
            else:
                self.fc1_bias = init_params.get("fc1_no_gate_bias")
        self.fc2_weight = init_params.get("fc2_weight")

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
            hidden_size=INPUT_SIZE,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.args.tensor_parallel,
            add_bias_linear=self.has_bias,
            gated_linear_unit=self.gated_linear_unit,
            hidden_act="silu",
            num_layers=1,
            compute_dtype='bf16',
            params_dtype='fp32',
        )

    def build_model(self):
        """Build MLP module"""
        net = build_module(
            get_mlp_module_spec(gated_linear_unit=self.config.gated_linear_unit),
            config=self.config,
            input_size=self.input_size,
            tp_group=self.model_comm_pgs.tp,
        )
        param_dict = {
            "linear_fc1.weight": self.fc1_weight,
            "linear_fc2.weight": self.fc2_weight,
        }
        if self.has_bias:
            param_dict["linear_fc1.bias"] = self.fc1_bias
            param_dict["linear_fc2.bias"] = self.fc2_bias
        self._load_weights(net, param_dict)
        return net

    def _load_weights(self, net, param_dict):
        """load weights for mlp module"""
        tp_group_size = self.args.tensor_parallel
        rank_id = get_rank()
        new_param_dict = {}

        def split(weight, split_axis=0):
            split_size = weight.shape[split_axis] // tp_group_size
            start = rank_id * split_size
            stop = (rank_id + 1) * split_size
            return weight[start:stop] if split_axis == 0 else weight[:, start:stop]

        w_fc1 = param_dict["linear_fc1.weight"]
        w_fc2 = param_dict["linear_fc2.weight"]
        if self.gated_linear_unit:
            w_gate = w_fc1[:self.ffn_hidden_size, :]
            w_hidden = w_fc1[self.ffn_hidden_size:, :]
            w_gate_shard = split(w_gate)
            w_hidden_shard = split(w_hidden)
            w_fc1_shard = np.concatenate([w_gate_shard, w_hidden_shard], axis=0)
        else:
            w_fc1_shard = split(w_fc1)
        w_fc2_shard = split(w_fc2, split_axis=1)
        new_param_dict["linear_fc1.weight"] = Parameter(w_fc1_shard)
        new_param_dict["linear_fc2.weight"] = Parameter(w_fc2_shard)

        if self.has_bias:
            w_fc1_bias = param_dict["linear_fc1.bias"]
            w_fc2_bias = param_dict["linear_fc2.bias"]
            w_fc1_bias = w_fc1_bias.reshape(
                self.ffn_hidden_size * 2 if self.gated_linear_unit else self.ffn_hidden_size, -1)
            w_fc1_bias_shard = split(w_fc1_bias)
            w_fc1_bias_shard = w_fc1_bias_shard.reshape(-1)
            w_fc2_bias_shard = w_fc2_bias
            new_param_dict["linear_fc1.bias"] = Parameter(w_fc1_bias_shard)
            new_param_dict["linear_fc2.bias"] = Parameter(w_fc2_bias_shard)

        ms.load_param_into_net(net, new_param_dict)

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        output = net(self.input)
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.asnumpy().astype(np.float16) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run MLP test")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--ffn_hidden_size", type=int, default=32)
    parser.add_argument("--has_bias", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--gated_linear_unit", type=lambda x: x.lower() == "true", default=True)
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
    runner = MLPRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
