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

"""Component that check gradient against numeric with respect to inputs."""

from ...components.icomponent import IExectorComponent
from ...utils.check_gradient import check_gradient, OperationGradChecker
from ...utils.config_util import get_grad_checking_options


class CheckGradientWrtInputsEC(IExectorComponent):
    """
    Check gradient against numeric with respect to inputs, execute and verify.

    Examples:
        'block': BertAttentionQueryKeyMul(batch_size=1,
                                          from_tensor_width=1024,
                                          to_tensor_width=1024,
                                          from_seq_length=128,
                                          to_seq_length=128,
                                          num_attention_heads=16,
                                          size_per_head=64,
                                          query_act=None,
                                          key_act=None,
                                          initializer_range=0.02)
    """

    def __call__(self):
        f, args, delta, max_error, input_selector, output_selector, \
        sampling_times, reduce_output = get_grad_checking_options(self.function, self.inputs)
        check_gradient(f, *args, delta=delta, max_error=max_error, grad_checker_class=OperationGradChecker,
                       input_selector=input_selector, output_selector=output_selector, sampling_times=sampling_times,
                       reduce_output=reduce_output)
