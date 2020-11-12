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
"""hub config."""
from src.models import get_generator

def create_network(name, *args, **kwargs):
    if name == "cyclegan":
        G_A = get_generator(*args, **kwargs)
        G_B = get_generator(*args, **kwargs)
        # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
        # Use real mean and varance rather than moving_men and moving_varance in BatchNorm2d
        G_A.set_train(True)
        G_B.set_train(True)
        return G_A, G_B
    raise NotImplementedError(f"{name} is not implemented in the repo")
