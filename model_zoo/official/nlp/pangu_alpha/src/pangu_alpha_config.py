# Copyright 2021 Huawei Technologies Co., Ltd
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
network config setting
"""
import mindspore.common.dtype as mstype


class PANGUALPHAConfig:
    """
    PANGUALPHA config class which defines the model size
    """
    def __init__(self,
                 data_parallel_num,
                 model_parallel_num,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=50257,
                 embedding_size=768,
                 num_layers=12,
                 num_heads=12,
                 expand_ratio=4,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 word_emb_dp=True,
                 stage_num=16,
                 eod_reset=True,
                 micro_size=32,
                 load_ckpt_path=None,
                 use_top_query_attention=True,
                 param_init_type=mstype.float32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        # The expand ratio of feature size in FFN
        self.expand_ratio = expand_ratio
        # Use post-layernorm or pre-layernrom, default:pre-layernorm
        self.post_layernorm_residual = post_layernorm_residual
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype
        # Whether use incremental inference
        self.use_past = use_past
        self.dp = data_parallel_num
        self.mp = model_parallel_num
        self.stage_num = stage_num
        self.micro_size = micro_size
        self.word_emb_dp = word_emb_dp
        self.eod_reset = eod_reset
        # Used for loading embedding tables
        self.load_ckpt_path = load_ckpt_path
        self.use_top_query_attention = use_top_query_attention
        self.param_init_type = param_init_type

    def __str__(self):
        info = "[PANGUALPHAConfig]" + '===' * 10 + '\n'
        for k, v in self.__dict__.items():
            var_info = "{}:{}\n".format(k, v)
            info += var_info
        info += '=' * 10
        return info

def set_parse(args_opt):
    r"""
       Set config according to the mode
    """
    if args_opt.mode == "200B":
        args_opt.embedding_size = 16384
        args_opt.num_layers = 64
        args_opt.num_heads = 128
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1
        args_opt.word_emb_dp = 0
        if args_opt.run_type == "train":
            args_opt.start_lr = 6e-5
            args_opt.end_lr = 6e-6
            args_opt.stage_num = 16
            args_opt.micro_size = 32
            args_opt.op_level_model_parallel_num = 16
            if args_opt.optimizer_shard == 1:
                args_opt.op_level_model_parallel_num = 8
        elif args_opt.run_type == "predict":
            args_opt.stage_num = 4
            args_opt.micro_size = 1
            args_opt.op_level_model_parallel_num = 16
            if args_opt.optimizer_shard == 1:
                args_opt.op_level_model_parallel_num = 8
    elif args_opt.mode == "13B":
        args_opt.embedding_size = 5120
        args_opt.num_layers = 40
        args_opt.num_heads = 40
        args_opt.word_emb_dp = 1
        args_opt.op_level_model_parallel_num = 8
        if args_opt.run_type == "train":
            args_opt.start_lr = 5e-5
            args_opt.end_lr = 1e-6
            args_opt.optimizer_shard = 1
            args_opt.full_batch = 0
            if args_opt.per_batch_size == 0:
                args_opt.per_batch_size = 8
            if args_opt.stage_num > 1:
                args_opt.word_emb_dp = 0
        elif args_opt.run_type == "predict":
            args_opt.stage_num = 1
            args_opt.micro_size = 1
            if args_opt.per_batch_size == 0:
                args_opt.per_batch_size = 1
    elif args_opt.mode == "2.6B":
        args_opt.embedding_size = 2560
        args_opt.num_layers = 32
        args_opt.num_heads = 32
        args_opt.op_level_model_parallel_num = 8
        if args_opt.run_type == "train":
            args_opt.start_lr = 1e-4
            args_opt.end_lr = 1e-6
            args_opt.optimizer_shard = 1
            args_opt.full_batch = 0
            if args_opt.per_batch_size == 0:
                args_opt.per_batch_size = 16
            if args_opt.stage_num > 1:
                args_opt.word_emb_dp = 0
        elif args_opt.run_type == "predict":
            args_opt.stage_num = 1
            args_opt.micro_size = 1
            if args_opt.per_batch_size == 0:
                args_opt.per_batch_size = 1
