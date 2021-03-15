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
One Hop Model.

"""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net


class Model(nn.Cell):
    """mlp model"""
    def __init__(self):
        super(Model, self).__init__()
        self.tanh_0 = nn.Tanh()
        self.dense_1 = nn.Dense(in_channels=768, out_channels=1, has_bias=True)

    def construct(self, x):
        """construct function"""
        opt_tanh_0 = self.tanh_0(x)
        opt_dense_1 = self.dense_1(opt_tanh_0)
        return opt_dense_1


class OneHopBert(nn.Cell):
    """onehop model"""
    def __init__(self, config, network):
        super(OneHopBert, self).__init__(auto_prefix=False)
        self.network = network
        self.mlp = Model()
        param_dict = load_checkpoint(config.onehop_mlp_path)
        load_param_into_net(self.mlp, param_dict)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  token_type_id,
                  input_mask):
        """construct function"""
        out = self.network(input_ids, token_type_id, input_mask)
        out = self.mlp(out)
        out = self.cast(out, mstype.float32)
        return out
