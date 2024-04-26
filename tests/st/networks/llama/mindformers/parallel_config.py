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
"""Parallel Config Init."""
from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
from mindspore.parallel._cost_model_context import _set_rp_matmul_mem_coef

default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


def build_parallel_config(config):
    """
    Build context config.

    Args:
            - config (Union[MindFormerConfig, Dict]) - Input config.

    Returns:
            - config (Union[MindFormerConfig, Dict]) - Output config,
                with its moe_config, recompute_config and parallel_config
                are initialized from dict or assigned with a default one.
    """

    if config.recompute_config:
        if not isinstance(config.recompute_config, TransformerRecomputeConfig):
            config.recompute_config = TransformerRecomputeConfig(**config.recompute_config)
    else:
        config.recompute_config = default_recompute_config
    if config.parallel_config:
        if not isinstance(config.parallel_config, TransformerOpParallelConfig):
            if config.parallel_config.pipeline_stage > 1:
                config.parallel_config.vocab_emb_dp = False
            _set_rp_matmul_mem_coef(config.parallel_config.pop('mem_coeff', 0.25))
            config.parallel_config = TransformerOpParallelConfig(recompute=config.recompute_config,
                                                                 **config.parallel_config)
    else:
        config.parallel_config = default_parallel_config
