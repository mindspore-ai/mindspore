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
"""Llama2 Base Model."""

import os
import sys

workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.models.llama import LlamaForCausalLM, LlamaConfig
from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
from mindformers import CosineWithWarmUpLR


NUM_LAYERS = 16
NUM_HEADS = 4
HIDDEN_SIZE = 512
SEQ_LENGTH = 4096

RECOMPUTE_CONFIG = TransformerRecomputeConfig(
    recompute=False,
    select_recompute=False,
    parallel_optimizer_comm_recompute=False,
    mp_comm_recompute=True,
    recompute_slice_activation=False
)

PARALLEL_CONFIG = TransformerOpParallelConfig(
    data_parallel=2,
    model_parallel=2,
    expert_parallel=1,
    pipeline_stage=1,
    micro_batch_num=2,
    recompute=RECOMPUTE_CONFIG,
    use_seq_parallel=False,
    gradient_aggregation_group=4,
    vocab_emb_dp=True,
)

BASE_CONFIG = {
    'trainer': {
        'type': 'CausalLanguageModelingTrainer',
        'model_name': 'llama2'
    },
    'train_dataset': {
        'batch_size': 2
    },
    'train_dataset_task': {},
    'micro_batch_interleave_num': 1,
    'use_parallel': True,
    'use_legacy_format': True,
    'parallel': {
        'parallel_mode': 1,  # 0-data parallel, 1-semi-auto parallel, 2-auto parallel, 3-hybrid parallel
        'gradients_mean': False,
        'enable_alltoall': False,
        'full_batch': True,
        'search_mode': "sharding_propagation",
        'enable_parallel_optimizer': True,
        'strategy_ckpt_save_file': r"./ckpt_strategy.ckpt",
        'parallel_optimizer_config': {
            'gradient_accumulation_shard': False,
            'parallel_optimizer_threshold': 64,
        }
    },
    'parallel_config': {
        'data_parallel': 2,
        'model_parallel': 2,
        'pipeline_stage': 1,
        'use_seq_parallel': False,
        'micro_batch_num': 2,
        'vocab_emb_dp': True,
        'gradient_aggregation_group': 4,
    },
    'runner_config': {
        'epochs': 1,
        'batch_size': 2,
        'sink_mode': True,
        'sink_size': 1
    },
    'monitor_config': {},
    # optimizer
    'optimizer': {
        'type': "AdamW",
        'betas': [0.9, 0.999],
        'eps': 1.e-8,
        'learning_rate': 5.e-5,
    },
    'context': {
        'mode': 0,  # 0--Graph Mode; 1--Pynative Mode
        'device_target': "Ascend"
    },
    # lr schedule
    'lr_schedule': {
        'type': CosineWithWarmUpLR,
        'learning_rate': 5.e-5,
        'lr_end': 1.e-6,
        'total_steps': 64
    },
    'runner_wrapper': {
        'type': 'MFTrainOneStepCell',
        'scale_sense': {
            'type': 'DynamicLossScaleUpdateCell',
            'loss_scale_value': 65536,
            'scale_factor': 1,
            'scale_window': 1000
        },
        'use_clip_grad': True,
    },
    'model': {
        'model_config': {
            'type': LlamaConfig,
            'offset': 0,
            'batch_size': 1,
            'seq_length': 4096,
            'hidden_size': 512,
            'num_layers': 16,
            'num_heads': 4,
            'use_past': False,
            'compute_dtype': "float16",
            'param_init_type': "float16",
            'softmax_compute_type': "float16"
        }
    },
    'callbacks': [
        {
            'type': 'MFLossMonitor'
        },
        {
            'type': 'CheckpointMonitor',
            'prefix': "llama2_7b",
            'save_checkpoint_steps': 32,
            'integrated_save': False,
            'async_save': False,
            'checkpoint_format': 'safetensors',
            'remove_redundancy': True,
            'keep_checkpoint_max': 8,
            'directory': './output/test_resume_parallel'
        }
    ]
}


def get_config_dict():
    """Get config dict."""
    return BASE_CONFIG


def get_model_config():
    """get instanced model config."""
    return LlamaConfig(num_layers=NUM_LAYERS, seq_length=SEQ_LENGTH,
                       num_heads=NUM_HEADS, hidden_size=HIDDEN_SIZE,
                       parallel_config=PARALLEL_CONFIG)


def get_model(config):
    """Get instanced model."""
    return LlamaForCausalLM(config)
