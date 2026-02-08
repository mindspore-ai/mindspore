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
"""Run ColumnParallelLinear accuracy test with configurable parameters via args"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

import mindspore as ms
from mindspore.communication import init, get_rank

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups
from data_gen_utils import get_init_params
SCRIPT_DIR = Path(__file__).parent.resolve()


class ColumnParallelLinearRunner:
    """Class to manage ColumnParallelLinear model and weights"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.has_bias = self.args.bias
        self.skip_bias_add = self.args.skip_bias_add
        self.skip_weight_param_allocation = self.args.skip_weight_param_allocation
        self.use_weight_tensor = self.args.use_weight_tensor

        self.input_size = self.args.input_size
        self.output_size = self.args.output_size
        self.compute_dtype = ms.bfloat16
        self.param_init_dtype = ms.bfloat16

        init_params = get_init_params(self.args.input_size, self.args.output_size)

        self.inputs = ms.Tensor(init_params.get("inputs"), dtype=ms.bfloat16)
        self.weight_tensor_input = ms.Tensor(init_params.get("weight_tensor_input"), dtype=ms.bfloat16)
        self.net_weight = init_params.get("weight")
        self.net_bias = init_params.get("bias")

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
            compute_dtype='bf16',
            num_layers=1,
            num_attention_heads=self.args.tensor_parallel,
        )

    def build_model(self):
        """Build and initialize ColumnParallelLinear model"""
        net = ColumnParallelLinear(
            input_size=self.input_size,
            output_size=self.output_size,
            config=self.config,
            skip_bias_add=self.skip_bias_add,
            skip_weight_param_allocation=self.skip_weight_param_allocation,
            compute_dtype=self.compute_dtype,
            gather_output=True,
            transpose_b=True,
            bias=self.has_bias,
            tp_group=self.model_comm_pgs.tp,
        )

        param_dict = {
            "weight": ms.Parameter(self.net_weight),
            "bias": ms.Parameter(self.net_bias)
            }

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

        weight = param_dict["weight"]
        weight_shard = split(weight, split_axis=0)
        new_param_dict["weight"] = ms.Parameter(weight_shard)

        bias = param_dict["bias"]
        bias_shard = split(bias)
        new_param_dict["bias"] = ms.Parameter(bias_shard)

        ms.load_param_into_net(net, new_param_dict)

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        weight_tensor_input = None
        if self.use_weight_tensor:
            weight_tensor_input = self.weight_tensor_input

        output = net(self.inputs, weight_tensor_input)
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.asnumpy().astype(np.float32) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run ColumnParallelLinear test")
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--output_size", type=int, default=32)
    parser.add_argument("--bias", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--skip_bias_add", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--skip_weight_param_allocation", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_weight_tensor", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    jit_level = "O0"
    infer_boost = "on"
    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": jit_level, "infer_boost": infer_boost})
    ms.set_seed(42)

    # Prepare input
    runner = ColumnParallelLinearRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
