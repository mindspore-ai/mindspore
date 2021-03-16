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
"""get_pretrained_model"""
import torch
from mindspore import Tensor, save_checkpoint


def torch2ms(pth_path, ckpt_path):
    """torch2ms"""
    pretrained_dict = torch.load(pth_path)
    print('--------------------pretrained keys------------------------')
    for k in pretrained_dict:
        print(k)

    print('---------------------torch2ms keys-----------------------')
    new_params = []
    for k, v in pretrained_dict.items():
        if 'fc' in k:
            continue
        if 'bn' in k or 'downsample.1' in k:
            k = k.replace('running_mean', 'moving_mean')
            k = k.replace('running_var', 'moving_variance')
            k = k.replace('weight', 'gamma')
            k = k.replace('bias', 'beta')
        k = 'network.resnet.' + k
        print(k)
        param_dict = {'name': k, 'data': Tensor(v.detach().numpy())}
        new_params.append(param_dict)
    save_checkpoint(new_params, ckpt_path)


if __name__ == '__main__':
    pth = "./resnet101-5d3b4d8f.pth"
    ckpt = "./resnet.ckpt"
    torch2ms(pth, ckpt)
