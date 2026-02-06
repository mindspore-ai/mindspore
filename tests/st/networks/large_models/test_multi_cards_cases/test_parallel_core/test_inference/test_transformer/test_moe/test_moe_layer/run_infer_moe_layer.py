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
"""Run mcore MOE parallel UT of inference with different parallel configs."""
import argparse
import os
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import Parameter

from run_infer_moe import MoERunner

SCRIPT_DIR = Path(__file__).parent.resolve()


class MoELayerRunner(MoERunner):
    """Class to manage MoELayer module"""

    def _load_weights(self, net, param_dict):
        """load weights for moe module"""
        new_param_dict = {}

        def split_tp_dp(weight, split_axis=0):
            split_size = weight.shape[split_axis] // self.global_group_size
            start = self.tp_dp_rank * split_size
            stop = (self.tp_dp_rank + 1) * split_size
            if split_axis == 0:
                return weight[start:stop]
            if split_axis == 1:
                return weight[:, start:stop]
            return weight[:, :, start:stop]

        def split_moe_tp(weight, split_axis=0):
            dim_size = weight.shape[split_axis]
            split_size = dim_size // self.moe_tp_group_size
            start = self.moe_tp_rank * split_size
            stop = (self.moe_tp_rank + 1) * split_size
            if split_axis == 1:
                return weight[self.ep_start:self.ep_stop, start:stop, :]
            return weight[self.ep_start:self.ep_stop, :, start:stop]

        # shared expert weight
        shared_expert_fc1_w = param_dict["shared_experts.linear_fc1.weight"]
        shared_expert_fc2_w = param_dict["shared_experts.linear_fc2.weight"]
        shared_expert_w_gate = shared_expert_fc1_w[:self.moe_intermediate_size, :]
        shared_expert_w_hidden = shared_expert_fc1_w[self.moe_intermediate_size:, :]
        if not self.config.use_alltoall:
            shared_expert_w_gate_shard = split_tp_dp(shared_expert_w_gate)
            shared_expert_w_hidden_shard = split_tp_dp(shared_expert_w_hidden)
            shared_expert_w_fc2_shard = split_tp_dp(shared_expert_fc2_w, split_axis=1)
        else:
            shared_expert_w_gate_shard = shared_expert_w_gate
            shared_expert_w_hidden_shard = shared_expert_w_hidden
            shared_expert_w_fc2_shard = shared_expert_fc2_w
        shared_expert_w_fc1_shard = np.concatenate([shared_expert_w_gate_shard, shared_expert_w_hidden_shard], axis=0)
        print(f"shared_expert_w_fc1_shard: {shared_expert_w_fc1_shard.shape}")
        print(f"shared_expert_w_fc2_shard: {shared_expert_w_fc2_shard.shape}")
        new_param_dict["shared_experts.linear_fc1.weight"] = Parameter(shared_expert_w_fc1_shard)
        new_param_dict["shared_experts.linear_fc2.weight"] = Parameter(shared_expert_w_fc2_shard)

        # router
        router_weight = param_dict["router.weight"]
        router_expert_bias = param_dict["router.expert_bias"]
        if self.ep_group_size > 1 and not self.config.use_alltoall:
            expert_idx = list(range(router_weight.shape[0]))
            in_start_expert_idx = self.num_experts // self.ep_group_size * self.ep_rank
            expert_idx = expert_idx[in_start_expert_idx:] + expert_idx[:in_start_expert_idx]
            router_weight = router_weight[expert_idx]
            router_expert_bias = router_expert_bias[expert_idx]
        new_param_dict["router.weight"] = Parameter(router_weight)
        new_param_dict["router.expert_bias"] = Parameter(router_expert_bias)

        # expert
        expert_fc1_w = param_dict["experts.weight1"]
        expert_fc2_w = param_dict["experts.weight2"]
        expert_w_gate = expert_fc1_w[:, :, :self.moe_intermediate_size]
        expert_w_hidden = expert_fc1_w[:, :, self.moe_intermediate_size:]
        expert_w_gate_shard = split_moe_tp(expert_w_gate, split_axis=2)
        expert_w_hidden_shard = split_moe_tp(expert_w_hidden, split_axis=2)
        print(f"expert_w_gate_shard: {expert_w_gate_shard.shape}")
        print(f"expert_w_hidden_shard: {expert_w_hidden_shard.shape}")
        expert_w_fc1_shard = np.concatenate([expert_w_gate_shard, expert_w_hidden_shard], axis=2)
        expert_w_fc2_shard = split_moe_tp(expert_fc2_w, split_axis=1)
        new_param_dict["experts.weight1"] = Parameter(expert_w_fc1_shard)
        new_param_dict["experts.weight2"] = Parameter(expert_w_fc2_shard)
        ms.load_param_into_net(net, new_param_dict)


def main():
    parser = argparse.ArgumentParser(description="Run MoELayer test")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_shared_experts", type=int, default=1)
    parser.add_argument("--routed_scaling_factor", type=float, default=32)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--n_group", type=int, default=2)
    parser.add_argument("--topk_group", type=int, default=2)
    parser.add_argument("--moe_intermediate_size", type=int, default=8)
    parser.add_argument("--moe_shared_expert_intermediate_size", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--expert_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    jit_level = "O0"
    infer_boost = "on"

    os.environ["RUN_MODE"] = "predict"
    os.environ["HCCL_DETERMINISTIC"] = "True"
    os.environ["MS_ENABLE_LCCL"] = "off"
    os.environ["CUSTOM_MATMUL_SHUFFLE"] = "off"

    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": jit_level, "infer_boost": infer_boost},
                   graph_kernel_flags="--disable_pass=matmul_split_fusion")
    seed_value = 2025
    ms.set_seed(seed_value)
    np.random.seed(seed_value)

    # Prepare input
    runner = MoELayerRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
