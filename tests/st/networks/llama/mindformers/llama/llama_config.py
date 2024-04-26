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
"""Llama Config API."""
import copy
from typing import Optional, Union, Dict, Any

from mindspore._checkparam import args_type_check
from mindspore._c_expression.typing import Float, BFloat

from mindformers.llama_utils import convert_mstype
from mindformers.modules.transformer.moe import MoEConfig, default_moe_config
from mindformers.modules.transformer.op_parallel_config import default_transformer_config, \
    TransformerOpParallelConfig
from mindformers.llama_utils import ms_type_to_str
from mindformers.normal_config import DictConfig

__all__ = ['LlamaConfig']


class LlamaConfig:
    """
    LLaMA config class which defines the model size.

    Args:
        batch_size (Optional[int]): batch size for input data, use in predict.
        seq_length (Optional[int]): The sequence length of input_ids, default is 1024.
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the BERT model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        multiple_of (Optional[int]): Define SwiGLU hidden layer size multiples, default 256.
        n_kv_heads (Optional[int]): Define multi group head attention heads number, default None.
        ffn_dim_multiplier (Optional[int]): Define ffn layer dim multiples, default None.
        rms_norm_eps (Optional[float]): The epsilon value of the denominator. Default 1e-5.
        bos_token_id (Optional[int]): The id of the *beginning-of-sequence* token.
        eos_token_id (Optional[int]): The id of the *end-of-sequence* token.
        pad_token_id (Optional[int]): The id of the *padding* token.
        ignore_token_id (Optional[int]): The id of the *ignoring* token.
        compute_dtype (Optional[str]):
            Linear layer compute dtype, default is "float16".
        layernorm_compute_type (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_compute_type (Optional[str]):
            softmax compute dtype, default is "float32".
        rotary_dtype (Optional[str]):
            rope compute dtype, default is "float32".
        param_init_type (Optional[str]):
            parameter initial dtype, default is "float16".
        qkv_has_bias (Optional[bool]):
            Whether the Query, Key, and Value projection has bias.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        pretrain_seqlen(int): The pretrained model seq length, default 2048.
        extend_method(str): The extend method of seq length of inferencem,default None.
        compute_in_2d(bool): Whether compute in 2-dims tensor, default False.
        use_flash_attention(bool): Whether enable flash attention ops, default False.
        use_paged_attention(bool): Whether enable paged attention ops, default False.
        offset(int): Offset of transformer layer when set pipeline stage number.
        use_past_shard(bool): The configuration of kvcache parallel shard, default False.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        max_decode_length (`int`, *optional*, defaults to 1024):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        top_k (`int`, *optional*, defaults to 5):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        block_size (`int`, *optional*, defaults to 16):
            The maximum number of tokens in one block can have when using paged attention.
        num_blocks (`int`, *optional*, defaults to 512):
            The maximum number of blocks when using paged attention.

        Returns:
            Class, LlamaConfig.
    """

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig))
    def __init__(self,
                 batch_size: int = 1,
                 seq_length: int = 2048,
                 hidden_size: int = 4096,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 n_kv_heads: Optional[int] = None,
                 max_position_embedding: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 vocab_size: int = 32000,  # defined later by tokenizer
                 multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
                 ffn_dim_multiplier: Optional[int] = None,
                 rms_norm_eps: float = 1e-5,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 pad_token_id: int = 0,
                 ignore_token_id: int = -100,
                 theta: float = 10000.0,
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 rotary_dtype: str = "float32",
                 param_init_type: str = "float16",
                 qkv_has_bias: bool = False,
                 qkv_concat: bool = False,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 use_past: bool = False,
                 pretrain_seqlen=None,
                 compute_in_2d=None,
                 use_past_shard=None,
                 extend_method: str = "None",
                 scaling_factor: float = 1.0,
                 is_dynamic: bool = False,
                 use_kvcache_op: bool = False,
                 is_flexible_shape: bool = False,
                 use_rope_slice: bool = False,
                 use_flash_attention: bool = False,
                 use_paged_attention: bool = False,
                 use_prompt_flash_attention: bool = False,
                 use_incre_flash_attention: bool = False,
                 fine_grain_interleave: int = 1,
                 offset: int = 0,
                 checkpoint_name_or_path: str = "",
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 1024,
                 block_size: int = 16,
                 num_blocks: int = 512,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 **kwargs):
        # super(LlamaConfig, self).__init__(**kwargs)
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        self._commit_hash = kwargs.pop("_commit_hash", None)

        self.checkpoint_name_or_path = kwargs.pop("checkpoint_name_or_path", None)

        # version info
        self.mindformers_version = kwargs.pop("mindformers_version", None)
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)

        # general config
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", None)

        # generation config
        self.is_sample_acceleration = kwargs.pop("is_sample_acceleration", None)

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err

        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embedding = max_position_embedding if max_position_embedding else seq_length
        self.intermediate_size = intermediate_size
        self.multiple_of = multiple_of
        self.n_kv_heads = n_kv_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rms_norm_eps = rms_norm_eps
        self.qkv_concat = qkv_concat
        self.param_init_type = convert_mstype(param_init_type)
        self.qkv_has_bias = qkv_has_bias
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.parallel_config = parallel_config
        self.moe_config = moe_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id
        self.use_past = use_past
        if pretrain_seqlen is not None:
            self.pretrain_seqlen = pretrain_seqlen
        if compute_in_2d is not None:
            self.compute_in_2d = compute_in_2d
        if use_past_shard is not None:
            self.use_past_shard = use_past_shard
        self.extend_method = extend_method
        self.scaling_factor = scaling_factor
        self.is_dynamic = is_dynamic
        self.use_kvcache_op = use_kvcache_op
        self.is_flexible_shape = is_flexible_shape
        self.use_rope_slice = use_rope_slice
        self.use_flash_attention = use_flash_attention
        self.use_paged_attention = use_paged_attention
        self.use_prompt_flash_attention = use_prompt_flash_attention
        self.use_incre_flash_attention = use_incre_flash_attention
        self.fine_grain_interleave = fine_grain_interleave
        self.offset = offset
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.theta = theta
        self.block_size = block_size
        self.num_blocks = num_blocks

    @property
    def name_or_path(self) -> str:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)


    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]
        self._to_dict_helper(output)

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, DictConfig):
                value = value.to_dict()

            output[key] = value

        self.dict_ms_dtype_to_str(output)
        return output

    def _to_dict_helper(self, output):
        if "parallel_config" in output:
            output["parallel_config"] = output["parallel_config"].to_dict()
        if "moe_config" in output:
            output["moe_config"] = output["moe_config"].to_dict()
        if "op_parallel_config" in output:
            output["op_parallel_config"] = output["op_parallel_config"].to_dict()
        if "embed_parallel_config" in output:
            output["embed_parallel_config"] = output["embed_parallel_config"].to_dict()

    def dict_ms_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *ms_dtype* key and if it's not None,
        converts ms.dtype to a string of just the type. For example, `ms.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        for k, v in d.items():
            if isinstance(v, (Float, BFloat)):
                d[k] = ms_type_to_str[v]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_ms_dtype_to_str(value)
