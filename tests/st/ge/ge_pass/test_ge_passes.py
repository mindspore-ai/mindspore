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
def test_avg_pool_grad():
    """
    Description: Auto-diff AvgPool in ge backend
    Expectation: success
    """
    utils.run_testcase('pass_avg_pool_grad')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dropout():
    """
    Description: run dropout and dropoutgrad in ge backend
    Expectation: success
    """
    utils.run_testcase('pass_dropout')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reduce_axis_update():
    """
    Description: test axis of reduce operator is empty
    Expectation: success
    """
    utils.run_testcase('pass_reduce_axis_update')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_convert_attr_to_input():
    """
    Description: test convert attr to input
    Expectation: success
    """
    utils.run_testcase('pass_convert_attr_to_input')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_convert_resize_nearest_neighbor_x_dtype():
    """
    Description: test convert ReszieNearestNeighborX dytpe
    Expectation: success
    """
    utils.run_testcase('pass_convert_resize_nearest_neighbor_x_dtype')

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_adaptive_max_pool2d_x_dtype():
    """
    Description: test AdaptiveMaxPool2DGeFusion dytpe
    Expectation: success
    """
    utils.run_testcase('pass_adaptive_max_pool2d')
