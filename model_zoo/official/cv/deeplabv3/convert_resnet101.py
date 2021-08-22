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
"""convert backbone resnet101"""
import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint


def torch2ms():
    pretrained_dict = torch.load('./resnet101-5d3b4d8f.pth')
    new_params = []

    for key, value in pretrained_dict.items():
        if not key.__contains__('fc'):
            if key.__contains__('bn'):
                key = key.replace('running_mean', 'moving_mean')
                key = key.replace('running_var', 'moving_variance')
                key = key.replace('weight', 'gamma')
                key = key.replace('bias', 'beta')
            param_dict = {'name': key, 'data': Tensor(value.detach().numpy())}
            new_params.append(param_dict)
    save_checkpoint(new_params, './resnet101-5d3b4d8f.ckpt')
    print("Convert resnet-101 completed!")


if __name__ == '__main__':
    torch2ms()
