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
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp


class AddCustomAclnnNet(Cell):
    def __init__(self, func, out_shape, bprop):
        super(AddCustomAclnnNet, self).__init__()
        aclnn_ref_info = CustomRegOp("aclnnAddCustom") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_add = ops.Custom(func, out_shape, lambda x, _: x, func_type="aot", bprop=bprop,
                                     reg_info=aclnn_ref_info)

    def construct(self, x, y):
        return self.custom_add(x, y)


def test_custom_op_lambda_infer():
    """
    Feature: Testing custom operator.
    Description: Testing the use of a lambda infer.
    Expectation: The test should pass without raising any exceptions.
    """
    AddCustomAclnnNet("aclnnAddCustom", lambda x, _: x, None)


def test_custom_op_cpp_infer():
    """
    Feature: Testing custom operator.
    Description: Testing the use of cpp infer.
    Expectation: The test should pass without raising any exceptions.
    """
    script_path, _ = os.path.split(__file__)
    func = script_path + '/infer_file/add_custom_infer.cc:aclnnAddCustom'
    AddCustomAclnnNet(func, None, None)


def test_custom_op_func_type_1():
    """
    Feature: Testing custom operator.
    Description: Checking the type of the 'func' parameter for custom operation.
    Expectation: A TypeError should be raised with a message indicating that 'func' must be of type str.
    """
    try:
        AddCustomAclnnNet(None, lambda x, _: x, None)
    except TypeError as e:
        assert "'func' must be of type str" in str(e)


def test_custom_op_func_type_2():
    """
    Feature: Testing custom operator.
    Description: This test checks if the 'func' parameter is correctly formatted as 'file_name:func_name'.
    Expectation: A TypeError should be raised with a message stating that 'func' should be in the format
                 'file_name:func_name'.
    """
    try:
        AddCustomAclnnNet("aclnnAddCustom", None, None)
    except TypeError as e:
        assert "'func' should be like 'file_name:func_name'" in str(e)


def test_custom_op_white_list():
    """
    Feature: Testing custom operator.
    Description: Testing the white list path for custom operation files.
    Expectation: A TypeError should be raised with a message indicating that the file path is not legal.
    """
    os.putenv('MS_CUSTOM_AOT_WHITE_LIST', '/tmp/white_list/')
    try:
        script_path, _ = os.path.split(__file__)
        func = script_path + '/infer_file/add_custom_infer.cc:aclnnAddCustom'
        AddCustomAclnnNet(func, None, None)
    except TypeError as e:
        assert "the legal path for the file is" in str(e)
