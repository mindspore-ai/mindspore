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

import os
import re
import pytest
import numpy as np
from mindspore import context, Tensor, nn

def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))

def find_last_ir_file(folder_path):
    last_ir_files = [last_ir for last_ir in os.listdir(folder_path) if 'anf_graph_after_build_df_graph' in last_ir]
    assert len(last_ir_files) == 1
    last_ir_file = os.path.join(folder_path, last_ir_files[0])
    with open((os.path.join(last_ir_file)), 'r') as f:
        content = f.read()
    return content


class TestConvNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3,
                              kernel_size=1, stride=1, has_bias=False,
                              weight_init='ones', pad_mode='same')
        self.bn = nn.BatchNorm2d(num_features=3)

    def construct(self, x):
        y = self.conv(x)
        out = self.bn(y)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ge_last_ir():
    """
    Feature: dump last ir for ge
    Description: dump last ir before ge with whole info
    Expectation: success
    """
    save_path = "./test_ge_last_ir"
    clean_all_ir_files(save_path)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=1, save_graphs_path=save_path)
    context.set_context(jit_level="O2")
    input_x = Tensor(np.random.randint(2, size=(1, 3, 2, 2)).astype((np.float32)))
    net = TestConvNet()
    _ = net(input_x)

    last_ir_content = find_last_ir_file(save_path)
    # check fullname with scope
    conv_op_num = re.findall('Conv2D-op', last_ir_content)
    assert len(conv_op_num) == 1
    # check associated code line info
    code_line_info = re.findall('test_ge_save_last_ir', last_ir_content)
    assert len(code_line_info) >= 1
