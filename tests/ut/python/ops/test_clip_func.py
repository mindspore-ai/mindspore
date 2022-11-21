# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test_clip_func """
import functools

import numpy as np
import mindspore.nn as nn
from mindspore import ops
import mindspore.context as context
from mindspore import Tensor
from tests.mindspore_test_framework.mindspore_test import mindspore_test
from tests.mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config
from tests.mindspore_test_framework.pipeline.forward.verify_exception \
    import pipeline_for_verify_exception_for_case_by_case_config

context.set_context(mode=(context.GRAPH_MODE))



class NetWorkClipByValue(nn.Cell):
    __doc__ = ' NetWorkClipByValue definition '

    def __init__(self):
        super(NetWorkClipByValue, self).__init__()
        self.clip_func = ops.clip_by_value

    def construct(self, x, min_value, max_value):
        return self.clip_func(x, min_value, max_value)


class NetWorkClipListByValue(nn.Cell):
    __doc__ = ' NetWorkClipListByValue definition '

    def __init__(self):
        super(NetWorkClipListByValue, self).__init__()
        self.clip_func = ops.clip_by_value

    def construct(self, x1, x2, x3, min_value, max_value):
        return self.clip_func([x1, x2, x3], min_value, max_value)


test_case_clip_func = [
    ('ClipbyValue_1', {
        'block': NetWorkClipByValue(),
        'desc_inputs': [Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        2,
                        8.0],
        'skip': ['backward']}),
    ('ClipbyValue_2', {
        'block': NetWorkClipByValue(),
        'desc_inputs': [Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        None,
                        8],
        'skip': ['backward']}),
    ('ClipbyValue_3', {
        'block': NetWorkClipByValue(),
        'desc_inputs': [Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        Tensor(np.array([2]).astype(np.float32)),
                        Tensor(np.array(8).astype(np.float32))],
        'skip': ['backward']}),
    ('ClipListbyValue_1', {
        'block': NetWorkClipListByValue(),
        'desc_inputs': [Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        Tensor(np.array(2).astype(np.float32)),
                        8],
        'skip': ['backward']}),
]


test_cases_for_verify_exception = [
    ('ClipByValueERR_1', {
        'block': (NetWorkClipByValue(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        Tensor(np.array([2, 3]).astype(np.float32)),
                        8]}),
    ('ClipByValueERR_2', {
        'block': (NetWorkClipByValue(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        '2',
                        8]}),
    ('ClipByValueERR_3', {
        'block': (NetWorkClipByValue(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(0, 10, [2, 3, 4]).astype(np.float32)),
                        2,
                        '8']}),
    ('ClipByValueERR_4', {
        'block': (NetWorkClipByValue(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones([1, 3, 3]).astype(np.float32)),
                        None,
                        None]}),
]


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    """
    Feature: test list of clip function
    Description: test case
    Expectation: success
    """
    context.set_context(mode=(context.GRAPH_MODE))
    return functools.reduce(lambda x, y: x + y, [test_case_clip_func])


@mindspore_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_check_exception():
    """
    Feature: test list getitem exception
    Description: test list getitem exception
    Expectation: throw errors
    """
    return test_cases_for_verify_exception
