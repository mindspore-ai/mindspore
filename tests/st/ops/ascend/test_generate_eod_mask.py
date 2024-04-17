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
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

class Net(nn.Cell):
    """
    Examples:

    1. flip bit on each step:

    mask = P.inner_ops.GenerateEodMask(n_pos=1,  # the elements position of the tensor
                             eod_token_id=14, # which bit of the element
                             n_step=[1], # which step of the training, only list supported
                             n_error_mode='circle' # specific or circle,
                             )

    2. flip bit on specific steps with circle, for example, 5:
    mask = P.inner_ops.GenerateEodMask(n_pos=1,
                             eod_token_id=14,
                             n_step=[5],
                             n_error_mode='circle'
                             )

    3.flip bit on specific steps, for example, 5,7,8:
    mask = P.inner_ops.GenerateEodMask(n_pos=1,
                             eod_token_id=14,
                             n_step=[5, 7, 8],
                             n_error_mode='specific'
                             )
    """
    def __init__(self, n_pos, eod_token_id, n_step, n_error_mode="specific"):
        super(Net, self).__init__()
        self.mask = P.inner_ops.GenerateEodMask(n_pos,
                                                eod_token_id=eod_token_id, n_step=n_step,
                                                n_error_mode=n_error_mode)

    def construct(self, tensor):
        return self.mask(tensor)

def run_generate_eod_mask_on_step(data, element_pos, bit_pos, n_step, n_error_mode='specific'):
    """
    Feature: Test GenerateEodMask.
    Description: Test multi dtype inputs
    Expectation: Successful graph compilation.
    """
    source_data = data.asnumpy()
    net = Net(n_pos=element_pos, eod_token_id=bit_pos, n_step=n_step, n_error_mode=n_error_mode)
    for i in range(10):
        out = net(data)
        print(f"loop {i} the output is:", out)
        if n_error_mode == "specific":
            if i in n_step:
                assert source_data[0][element_pos] == -out.asnumpy()[0][element_pos]
            else:
                assert (source_data == out.asnumpy()).all()
        else:
            if i != 0 and i % n_step[0] == 0:
                assert source_data[0][element_pos] == -out.asnumpy()[0][element_pos]
            else:
                assert (source_data == out.asnumpy()).all()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_generate_eod_mask_on_each_step():
    """
    Feature: Test bit flip on the first element on 14-th bits on each step in 10 steps
    Description: Test multi dtype inputs
    Expectation: Successful graph compilation.
    """
    test_data = Tensor([[0.1]], dtype=mindspore.float32)
    run_generate_eod_mask_on_step(test_data, element_pos=0, bit_pos=31, n_step=[1], n_error_mode='circle')
