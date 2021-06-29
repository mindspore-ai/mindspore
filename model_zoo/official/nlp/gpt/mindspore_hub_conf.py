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
"""hub config"""
from mindspore import dtype
from src.gpt import GPT
from src.utils import GPTConfig

def gpt_net(*args, **kwargs):
    return GPT(*args, **kwargs)

def create_network(name, *args, **kwargs):
    """
    create net work gpt
    """
    if name == "gpt":
        config = GPTConfig(batch_size=16,
                           seq_length=1024,
                           vocab_size=50257,
                           embedding_size=1024,
                           num_layers=24,
                           num_heads=16,
                           expand_ratio=4,
                           post_layernorm_residual=False,
                           dropout_rate=0.0,
                           compute_dtype=dtype.float16,
                           use_past=False)
        return gpt_net(config, *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
