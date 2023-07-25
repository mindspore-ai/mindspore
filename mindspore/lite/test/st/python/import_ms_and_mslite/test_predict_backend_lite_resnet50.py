# Copyright 2023 Huawei Technologies Co., Ltd
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
######################## LiteInfer test ########################

Note:
    To run this scripts, 'mindspore' and 'mindspore_lite' must be installed.
    mindspore_lite must be cloud inference version.
"""
import os

import numpy as np

import mindspore as ms
from mindspore import context
from lite_infer_predict_utils import predict_backend_lite, _get_max_index_from_res
from resnet import resnet50


# pylint: disable=I1101
os.environ['MSLITE_ENABLE_CLOUD_INFERENCE'] = "on"


def create_model():
    """
    create model.
    """
    net = resnet50(1001)
    net.set_train(False)
    ms_model = ms.Model(net)
    return ms_model


def test_predict_backend_lite_resnet():
    """
    Feature: test LiteInfer predict.
    Description: test LiteInfer predict.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    fake_input = ms.Tensor(np.ones((1, 3, 224, 224)).astype(np.float32))

    model = create_model()
    res_lite, avg_t_lite = predict_backend_lite(model, fake_input)
    print("Predict using backend lite, res: ", _get_max_index_from_res(res_lite))
    print(f"Predict using backend lite, avg time: {avg_t_lite * 1000} ms")

    assert avg_t_lite > 0.0  # assert predict is ok
