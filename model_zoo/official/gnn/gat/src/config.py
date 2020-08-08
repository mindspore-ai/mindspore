# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Train configs for training gat"""


class GatConfig():
    lr = 0.005
    num_epochs = 200
    hid_units = [8]
    n_heads = [8, 1]
    early_stopping = 100
    l2_coeff = 0.0005
    attn_dropout = 0.6
    feature_dropout = 0.6
