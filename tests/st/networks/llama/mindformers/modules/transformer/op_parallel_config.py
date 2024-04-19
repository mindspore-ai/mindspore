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
Parallel Config for the Parallel Training
This is an experimental interface that is subject to change and/or deletion.
"""
from __future__ import absolute_import

from typing import Union

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import context
from mindspore import log as logger
from mindspore._checkparam import args_type_check
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
import mindspore.communication.management as D

__all__ = [
    "TransformerRecomputeConfig",
    "TransformerOpParallelConfig",
    "MoEParallelConfig",
    "OpParallelConfig",
    "EmbeddingOpParallelConfig"
]


class _Config:
    r""" A basic class of the configure"""

    def __str__(self):
        info = "[ParallelConfig]" + '\n'
        for k, v in self.__dict__.items():
            var_info = "{}:{}\n".format(k, v)
            info += var_info
        return info


class OpParallelConfig(_Config):
    def __init__(self, data_parallel=1, model_parallel=1, use_seq_parallel=False, select_recompute=False):
        Validator.check_positive_int(data_parallel, "data_parallel")
        Validator.check_positive_int(model_parallel, "model_parallel")
        Validator.check_bool(use_seq_parallel, "use_seq_parallel")
        Validator.check_bool(select_recompute, "select_recompute")
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.use_seq_parallel = use_seq_parallel
        self.select_recompute = select_recompute

    @property
    def data_parallel(self):
        return self._data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        Validator.check_positive_int(value, "data_parallel")
        self._data_parallel = value

    @property
    def model_parallel(self):
        return self._model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        Validator.check_positive_int(value, "model_parallel")
        self._model_parallel = value

    def __eq__(self, other) -> bool:
        return isinstance(other, OpParallelConfig) and (self.to_dict() == other.to_dict())

    def to_dict(self):
        """to dict"""
        config_dict = {
            'data_parallel': self.data_parallel,
            'model_parallel': self.model_parallel,
            'use_seq_parallel': self.use_seq_parallel,
            'select_recompute': self.select_recompute
        }
        return config_dict

    def to_diff_dict(self):
        config_dict = self.to_dict()
        default_dict = OpParallelConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict[k]:
                res_dict[k] = v
        return res_dict


class _PipeLineConfig(_Config):
    r"""
        PPConfig for the setting data parallel, model parallel

        Args:
            pipeline_stage (int): The number of the pipeline stages. Default: 1
            micro_batch_num (int): The model parallel way. Default: 1
        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> config=_PipeLineConfig(pipeline_stage=1, micro_batch_num=1)
    """

    def __init__(self, pipeline_stage=1, micro_batch_num=1):
        Validator.check_positive_int(pipeline_stage, "pipeline_stage")
        Validator.check_positive_int(micro_batch_num, "micro_batch_num")
        self.pipeline_stage = pipeline_stage
        self.micro_batch_num = micro_batch_num

    @property
    def pipeline_stage(self):
        return self._pipeline_stage

    @pipeline_stage.setter
    def pipeline_stage(self, value):
        Validator.check_positive_int(value, "pipeline_stage")
        self._pipeline_stage = value
        context.set_auto_parallel_context(pipeline_stages=value)

    @property
    def micro_batch_num(self):
        return self._micro_batch_num

    @micro_batch_num.setter
    def micro_batch_num(self, value):
        Validator.check_positive_int(value, "micro_batch_num")
        self._micro_batch_num = value


class TransformerRecomputeConfig(_Config):
    def __init__(self, recompute=False, select_recompute=False,
                 parallel_optimizer_comm_recompute=False,
                 mp_comm_recompute=True, recompute_slice_activation=False):
        Validator.check_bool(recompute, "recompute")
        Validator.check_bool(parallel_optimizer_comm_recompute, "parallel_optimizer_comm_recompute")
        Validator.check_bool(mp_comm_recompute, "mp_comm_recompute")
        Validator.check_bool(select_recompute, "select_recompute")
        Validator.check_bool(recompute_slice_activation, "recompute_slice_activation")
        self._recompute = recompute
        self._select_recompute = select_recompute
        self._parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute
        self._mp_comm_recompute = mp_comm_recompute
        self._recompute_slice_activation = recompute_slice_activation

    @property
    def recompute(self):
        return self._recompute

    @recompute.setter
    def recompute(self, value):
        Validator.check_bool(value, "recompute")
        self._recompute = value

    @property
    def select_recompute(self):
        return self._select_recompute

    @select_recompute.setter
    def select_recompute(self, value):
        Validator.check_bool(value, "select_recompute")
        self._select_recompute = value

    @property
    def parallel_optimizer_comm_recompute(self):
        return self._parallel_optimizer_comm_recompute

    @parallel_optimizer_comm_recompute.setter
    def parallel_optimizer_comm_recompute(self, value):
        Validator.check_bool(value, "parallel_optimizer_comm_recompute")
        self._parallel_optimizer_comm_recompute = value

    @property
    def mp_comm_recompute(self):
        return self._mp_comm_recompute

    @mp_comm_recompute.setter
    def mp_comm_recompute(self, value):
        Validator.check_bool(value, "mp_comm_recompute")
        self._mp_comm_recompute = value

    @property
    def recompute_slice_activation(self):
        return self._recompute_slice_activation

    @recompute_slice_activation.setter
    def recompute_slice_activation(self, value):
        Validator.check_bool(value, "recompute_slice_activation")
        self._recompute_slice_activation = value

    def __eq__(self, other) -> bool:
        return isinstance(other, TransformerRecomputeConfig) and (self.to_dict() == other.to_dict())

    def to_diff_dict(self):
        config_dict = self.to_dict()
        default_dict = TransformerRecomputeConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict[k]:
                res_dict[k] = v
        return res_dict

    def to_dict(self):
        config_dict = {
            "recompute": self._recompute,
            "select_recompute": self._select_recompute,
            "parallel_optimizer_comm_recompute": self._parallel_optimizer_comm_recompute,
            "mp_comm_recompute": self._mp_comm_recompute,
            "recompute_slice_activation": self._recompute_slice_activation,
        }
        return config_dict


class EmbeddingOpParallelConfig(_Config):
    def __init__(self, data_parallel=1, model_parallel=1,
                 use_seq_parallel=False, select_recompute=False,
                 vocab_emb_dp=True):
        self._dp_mp_config = OpParallelConfig(data_parallel=data_parallel,
                                              use_seq_parallel=use_seq_parallel,
                                              model_parallel=model_parallel,
                                              select_recompute=select_recompute)
        Validator.check_bool(vocab_emb_dp, "vocab_emb_dp")
        self.vocab_emb_dp = vocab_emb_dp
        self.use_seq_parallel = use_seq_parallel
        self.select_recompute = select_recompute

    @property
    def data_parallel(self):
        return self._dp_mp_config.data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        self._dp_mp_config.data_parallel = value

    @property
    def model_parallel(self):
        return self._dp_mp_config.model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        self._dp_mp_config.model_parallel = value

    @property
    def vocab_emb_dp(self):
        return self._vocab_emb_dp

    @vocab_emb_dp.setter
    def vocab_emb_dp(self, value):
        Validator.check_bool(value, "vocab_emb_dp")
        self._vocab_emb_dp = value

    @property
    def dp_mp_config(self):
        return self._dp_mp_config

    def __eq__(self, other) -> bool:
        return isinstance(other, EmbeddingOpParallelConfig) and (self.to_dict() == other.to_dict())

    def to_diff_dict(self):
        config_dict = self.to_dict()
        default_dict = EmbeddingOpParallelConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict[k]:
                res_dict[k] = v
        return res_dict

    def to_dict(self):
        """to dict"""
        config_dict = {
            'data_parallel': self.data_parallel,
            'model_parallel': self.model_parallel,
            'select_recompute': self.select_recompute,
            'use_seq_parallel': self.use_seq_parallel,
            'vocab_emb_dp': self.vocab_emb_dp
        }
        return config_dict


class MoEParallelConfig(_Config):
    def __init__(self, data_parallel=1, model_parallel=1, expert_parallel=1,
                 use_seq_parallel=False, select_recompute=False):
        Validator.check_positive_int(data_parallel, "data_parallel")
        Validator.check_positive_int(model_parallel, "model_parallel")
        Validator.check_positive_int(expert_parallel, "expert_parallel")
        Validator.check_bool(use_seq_parallel, "use_seq_parallel")
        Validator.check_bool(select_recompute, "select_recompute")
        self._dpmp = OpParallelConfig(data_parallel=data_parallel,
                                      model_parallel=model_parallel,
                                      use_seq_parallel=use_seq_parallel,
                                      select_recompute=select_recompute)
        self.expert_parallel = expert_parallel
        self.use_seq_parallel = use_seq_parallel
        self.select_recompute = select_recompute

    @property
    def data_parallel(self):
        return self._dpmp.data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        Validator.check_positive_int(value, "data_parallel")
        self._dpmp.data_parallel = value

    @property
    def model_parallel(self):
        return self._dpmp.model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        Validator.check_positive_int(value, "model_parallel")
        self._dpmp.model_parallel = value

    @property
    def expert_parallel(self):
        return self._expert_parallel

    @expert_parallel.setter
    def expert_parallel(self, value):
        Validator.check_positive_int(value, "expert_parallel")
        self._expert_parallel = value

    @property
    def dpmp(self):
        """ Get the configuration for dpmp """
        return self._dpmp


class TransformerOpParallelConfig(_Config):
    @args_type_check(recompute=(TransformerRecomputeConfig, dict))
    def __init__(self, data_parallel=1, model_parallel=1, expert_parallel=1, pipeline_stage=1, micro_batch_num=1,
                 recompute: Union[TransformerRecomputeConfig, dict] = TransformerRecomputeConfig(),
                 use_seq_parallel=False, optimizer_shard=None, gradient_aggregation_group=4, vocab_emb_dp=True):
        if isinstance(recompute, dict):
            recompute = TransformerRecomputeConfig(**recompute)
        self.recompute = recompute
        self.select_recompute = recompute.select_recompute
        self.use_seq_parallel = use_seq_parallel
        self.optimizer_shard = optimizer_shard
        self.gradient_aggregation_group = gradient_aggregation_group
        self._embed_dp_mp_config = EmbeddingOpParallelConfig(
            data_parallel=data_parallel, model_parallel=model_parallel,
            vocab_emb_dp=vocab_emb_dp, use_seq_parallel=use_seq_parallel,
            select_recompute=recompute.select_recompute)
        self._pp_config = _PipeLineConfig(pipeline_stage=pipeline_stage, micro_batch_num=micro_batch_num)
        self._moe_config = MoEParallelConfig(
            data_parallel=data_parallel, model_parallel=model_parallel,
            select_recompute=recompute.select_recompute,
            expert_parallel=expert_parallel, use_seq_parallel=use_seq_parallel)

    def __eq__(self, other) -> bool:
        return isinstance(other, TransformerOpParallelConfig) and (self.to_dict() == other.to_dict())

    def to_diff_dict(self):
        config_dict = self.to_dict()
        default_dict = TransformerOpParallelConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict[k]:
                res_dict[k] = v
        if "recompute" in res_dict:
            res_dict["recompute"] = self.recompute.to_diff_dict()
        return res_dict

    def to_dict(self):
        """to dict"""
        config_dict = {
            'data_parallel': self.data_parallel,
            'model_parallel': self.model_parallel,
            'expert_parallel': self.expert_parallel,
            'pipeline_stage': self.pipeline_stage,
            'micro_batch_num': self.micro_batch_num,
            'use_seq_parallel': self.use_seq_parallel,
            'optimizer_shard': self.optimizer_shard,
            'gradient_aggregation_group': self.gradient_aggregation_group,
            'vocab_emb_dp': self.vocab_emb_dp,
            'recompute': self.recompute.to_dict()
        }
        return config_dict

    @property
    def recompute(self):
        return self._recompute

    @recompute.setter
    def recompute(self, value):
        if not isinstance(value, TransformerRecomputeConfig) and not isinstance(value, bool):
            raise TypeError(f"recompute must be a TransformerRecomputeConfig/bool, but got {type(value).__name__}.")
        if isinstance(value, bool):
            logger.warning(f"TransformerRecomputeConfig is recommended as the recompute configuration type.")
        self._recompute = value

    @property
    def vocab_emb_dp(self):
        return self._embed_dp_mp_config.vocab_emb_dp

    @vocab_emb_dp.setter
    def vocab_emb_dp(self, value):
        self._embed_dp_mp_config.vocab_emb_dp = value

    @property
    def gradient_aggregation_group(self):
        return self._gradient_aggregation_group

    @gradient_aggregation_group.setter
    def gradient_aggregation_group(self, value):
        Validator.check_positive_int(value, "gradient_aggregation_group")
        self._gradient_aggregation_group = value

    @property
    def micro_batch_num(self):
        return self._pp_config.micro_batch_num

    @micro_batch_num.setter
    def micro_batch_num(self, value):
        self._pp_config.micro_batch_num = value

    @property
    def model_parallel(self):
        return self._embed_dp_mp_config.model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        self._embed_dp_mp_config.model_parallel = value
        self._moe_config.model_parallel = value

    @property
    def data_parallel(self):
        return self._embed_dp_mp_config.data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        self._embed_dp_mp_config.data_parallel = value
        self._moe_config.data_parallel = value

    @property
    def expert_parallel(self):
        return self._moe_config.expert_parallel

    @expert_parallel.setter
    def expert_parallel(self, value):
        self._moe_config.expert_parallel = value

    @property
    def pipeline_stage(self):
        return self._pp_config.pipeline_stage

    @pipeline_stage.setter
    def pipeline_stage(self, value):
        self._pp_config.pipeline_stage = value

    @property
    def optimizer_shard(self):
        return self._optimizer_shard

    @optimizer_shard.setter
    def optimizer_shard(self, value):
        self._optimizer_shard = value
        if value:
            logger.warning("\"parallel_config.optimizer_shard\" is deprecated from MindFormers r0.7. It will not have "
                           "any effect. Please use \"parallel.enable_parallel_optimizer\" to turn on or off the "
                           "optimizer parallel.")

    @property
    def embedding_dp_mp_config(self):
        return self._embed_dp_mp_config

    @property
    def dp_mp_config(self):
        return self._embed_dp_mp_config.dp_mp_config

    @property
    def moe_parallel_config(self):
        return self._moe_config

default_dpmp_config = OpParallelConfig()
default_moeparallel_config = MoEParallelConfig()
default_transformer_config = TransformerOpParallelConfig()
default_embedding_parallel_config = EmbeddingOpParallelConfig()

def _check_config(config):
    """
       Check if micro_batch_num >= pipeline_stage
    """
    # the config pipeline_stage is same with context.pipeline_stage
    pipeline_stage = context.get_auto_parallel_context("pipeline_stages")
    if hasattr(config, 'pipeline_stage') and pipeline_stage != config.pipeline_stage:
        raise ValueError(
            f"The pipeline stage {pipeline_stage} in auto_parallel_context is not equal to the pipeline_stage "
            f"{config.pipeline_stage}"
            f" in the config.")

    # make sure the following is in auto parallel mode
    is_auto_parallel = _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
    if not is_auto_parallel:
        return

    device_num = D.get_group_size()
    optimizer_shard = context.get_auto_parallel_context("enable_parallel_optimizer")

    if config.data_parallel * config.model_parallel * pipeline_stage > device_num:
        raise ValueError(f"The product of the data parallel {config.data_parallel}, "
                         f"model parallel {config.model_parallel} "
                         f"pipeline stages {pipeline_stage} "
                         f"should be less than device_num {device_num}.")

    # the config optimizer_shard is same with context.optimizer_shard
    if hasattr(config, "optimizer_shard") and optimizer_shard and optimizer_shard != config.optimizer_shard:
        logger.warning(f"The optimizer shard {optimizer_shard} in auto_parallel_context is not equal to the"
                       f" optimizer_shard {config.optimizer_shard} in the OpParallelConfig. Please check the "
                       f"optimizer_shard to make them consistent.")
