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
import tests.st.ge.ge_test_utils as utils
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_broadcast_gradient_args():
    """
    Feature: for DynamicBroadcastGradientArgs op
    Description: inputs are two shapes
    Expectation: the result is correct
    """
    utils.run_testcase('broadcast_gradient_args')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv2d_backprop_filter():
    """
    Feature: for Conv2DBackpropFilter op
    Description: inputs are integers
    Expectation: the result is correct
    """
    utils.run_testcase('conv2d_backprop_filter')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv2d_backprop_input():
    """
    Feature: for Conv2DBackpropInput op
    Description: inputs are integers
    Expectation: the result is correct
    """
    utils.run_testcase('conv2d_backprop_input')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv2d_transpose():
    """
    Feature: for Conv2DTranspose op
    Description: inputs are integers
    Expectation: the result is correct
    """
    utils.run_testcase('conv2d_transpose')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_broadcast_to():
    """
    Feature: for DynamicBroadcastTo op
    Description: inputs are data and shape
    Expectation: the result is correct
    """
    utils.run_testcase('dynamic_broadcast_to')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unique():
    """
    Feature: for Unique op
    Description: inputs are integers
    Expectation: the result is correct
    """
    utils.run_testcase('unique')
