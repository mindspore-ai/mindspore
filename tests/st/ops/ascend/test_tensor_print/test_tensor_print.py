# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import os
import re
import pytest
import mindspore.context as context
from tensor_print_utils import run_net

# Defines the expected value of tensor printout, corresponding to different data types.
expect_scalar = {'Bool': 'True', 'UInt': '1', 'Int': '-1', 'Float16': '*.*******', 'Float32_64': '*.**'}
expect_array = {'Bool': '\n[[ True False]\n [False  True]]', 'UInt': '\n[[1 2 3]\n [4 5 6]]',
                'Int': '\n[[-1  2 -3]\n [-4  5 -6]]',
                'Float16': '\n[[ *.****e*** **.****e***  *.****e***]\n [ *.****e*** **.****e***  *.****e***]]',
                'Float32_64': '\n[[ *.********e*** **.********e***  *.********e***]\n ' \
                              '[ *.********e*** **.********e***  *.********e***]]'}

def get_expect_value(res):
    if res[0] == '[]':
        if res[1] == 'Bool':
            return expect_scalar.get('Bool')
        if res[1] in ['UInt8', 'UInt16', 'UInt32', 'UInt64']:
            return expect_scalar.get('UInt')
        if res[1] in ['Int8', 'Int16', 'Int32', 'Int64']:
            return expect_scalar.get('Int')
        if res[1] == 'Float16':
            return expect_scalar.get('Float16')
        if res[1] in ['Float32', 'Float64']:
            return expect_scalar.get('Float32_64')
    else:
        if res[1] == 'Bool':
            return expect_array.get('Bool')
        if res[1] in ['UInt8', 'UInt16', 'UInt32', 'UInt64']:
            return expect_array.get('UInt')
        if res[1] in ['Int8', 'Int16', 'Int32', 'Int64']:
            return expect_array.get('Int')
        if res[1] == 'Float16':
            return expect_array.get('Float16')
        if res[1] in ['Float32', 'Float64']:
            return expect_array.get('Float32_64')
    return 'None'

def num_to_asterisk(data):
    # Convert number and +/- to asterisk
    return re.sub(r'\d|\+|\-', '*', data.group())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_print():
    path = os.path.split(os.path.realpath(__file__))[0]
    cmd = f"python {path}/tensor_print_utils.py"
    lines = os.popen(cmd).readlines()
    data = ''.join(lines)
    result = re.findall(r'Tensor[(]shape=(.*?), dtype=(.*?), value=(.*?)[)]', data, re.DOTALL)
    assert (result != []), "Output does not meet the requirements."
    for res in result:
        assert (len(res) == 3), "Output does not meet the requirements."
        expect = get_expect_value(res)
        value = res[2]
        if value.find('.'):
            # Convert decimals to asterisks, such as 0.01 --> *.** and 1.0e+2 --> *.*e**
            value = re.sub(r'-?\d+\.\d+|e[\+|\-]\d+', num_to_asterisk, value, re.DOTALL)
        assert (repr(value) == repr(expect)), repr("output: " + value + ", expect: " + expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_print():
    """
    Feature: test tensor print.
    Description: print tensors.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_net()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_net()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_print_save_to_file():
    """
    Feature: test tensor print.
    Description: print tensors to files.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", print_file_path="output_data")
    run_net()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", print_file_path="output_data")
    run_net()
