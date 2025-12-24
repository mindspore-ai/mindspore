# Copyright 2025 Huawei Technologies Co., Ltd
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
Test custom ops utils for Ascend platform.
"""
from tests.mark_utils import arg_mark
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops.operations._custom_ops_utils import CustomInfoGenerator, CustomCodeGenerator


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_info_generator_add():
    """
    Feature: Custom op testcase
    Description: Test case for generating  info of Add op.
    Expectation: The generated registration info matches the expected structure.
    """
    info_generator = CustomInfoGenerator("Add")
    aclnn_api_types = info_generator.get_aclnn_api_types()
    expected_api_types = ['aclTensor*', 'aclTensor*', 'aclScalar*', 'aclTensor*', 'uint64_t*', 'aclOpExecutor**']
    assert aclnn_api_types == expected_api_types


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_info_generator_grouped_matmul():
    """
    Feature: Custom op testcase
    Description: Test case for generating info of MaxPool op.
    Expectation: The generated registration info matches the expected structure.
    """
    info_generator = CustomInfoGenerator("aclnnGroupedMatmul")
    aclnn_api_types = info_generator.get_aclnn_api_types()
    expected_api_types = ['aclTensorList*', 'aclTensorList*', 'aclTensorList*', 'aclTensorList*', 'aclTensorList*',
                          'aclTensorList*', 'aclTensorList*', 'aclIntArray*', 'int64_t', 'aclTensorList*',
                          'uint64_t*', 'aclOpExecutor**']
    assert aclnn_api_types == expected_api_types


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_info_generator_abs():
    """
    Feature: Custom op testcase
    Description: Test case for generating registration info of Abs op.
    Expectation: The generated registration info matches the expected structure.
    """
    reg_info_generator = CustomInfoGenerator("Abs")
    reg_info = reg_info_generator.generate_custom_reg_op()

    expected_reg_info = CustomRegOp("Abs") \
        .input(0, "x", "required") \
        .output(0, "y", "required") \
        .dtype_format(DataType.F16_Default, DataType.F16_Default) \
        .dtype_format(DataType.F32_Default, DataType.F32_Default) \
        .dtype_format(DataType.I32_Default, DataType.I32_Default) \
        .target("Ascend") \
        .get_op_info()

    assert reg_info == expected_reg_info

    aclnn_api_types = reg_info_generator.get_aclnn_api_types()
    expected_api_types = ['aclTensor*', 'aclTensor*', 'uint64_t*', 'aclOpExecutor**']
    assert aclnn_api_types == expected_api_types


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_info_generator_abs_grad():
    """
    Feature: Custom op testcase
    Description: Test case for generating registration info of AbsGrad op.
    Expectation: The generated registration info matches the expected structure.
    """
    reg_info_generator = CustomInfoGenerator("AbsGrad")
    reg_info = reg_info_generator.generate_custom_reg_op()
    expect_reg_info = CustomRegOp("AbsGrad") \
        .input(0, "y", "required") \
        .input(1, "dy", "required") \
        .output(0, "z", "required") \
        .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
        .dtype_format(DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_FracZ) \
        .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0) \
        .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
        .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
        .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ) \
        .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0) \
        .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
        .target("Ascend") \
        .get_op_info()
    assert reg_info == expect_reg_info


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_code_generator_add():
    """
    Feature: Custom op testcase
    Description: Test case for generating code of Add op.
    Expectation: The generated code matches the baseline code.
    """
    reg_info_generator = CustomInfoGenerator("Add")
    reg_info = reg_info_generator.generate_custom_reg_op()
    expect_reg_info = CustomRegOp("Add") \
        .input(0, "x1", "required") \
        .input(1, "x2", "required") \
        .output(0, "y", "required") \
        .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
        .target("Ascend") \
        .get_op_info()

    assert reg_info == expect_reg_info

    code_generator = CustomCodeGenerator()
    callback_func = code_generator.generate_callback_by_reg_info("Add", reg_info)
    with open("./callback_func/add_baseline.cc", "r", encoding="utf-8") as file:
        file_content = file.read()
        assert ''.join(file_content.split()) == ''.join(callback_func.split())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_code_generator_sort():
    """
    Feature: Custom op testcase
    Description: Test case for generating code of Sort op.
    Expectation: The generated code matches the baseline code.
    """
    reg_info_generator = CustomInfoGenerator("Sort")
    reg_info = reg_info_generator.generate_custom_reg_op()
    expect_reg_info = CustomRegOp("Sort") \
        .input(0, "x", "required") \
        .attr("axis", "optional", "int", "all") \
        .attr("descending", "optional", "bool", "all") \
        .attr("stable", "optional", "bool", "all") \
        .output(0, "y1", "required") \
        .output(1, "y2", "required") \
        .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
        .target("Ascend") \
        .get_op_info()

    assert reg_info == expect_reg_info

    code_generator = CustomCodeGenerator()
    callback_func = code_generator.generate_callback_by_reg_info("Sort", reg_info)
    with open("./callback_func/sort_baseline.cc", "r", encoding="utf-8") as file:
        file_content = file.read()
        assert ''.join(file_content.split()) == ''.join(callback_func.split())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_code_generator_mul():
    """
    Feature: Custom op testcase
    Description: Test case for generating code of Mul op.
    Expectation: The generated code matches the baseline code.
    """

    reg_info = CustomRegOp("aclnnMul") \
        .input(0, "x", "required") \
        .input(1, "y", "required") \
        .output(0, "z", "required") \
        .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
        .target("Ascend") \
        .get_op_info()

    code_generator = CustomCodeGenerator()
    callback_func = code_generator.generate_callback_by_reg_info("aclnnMul", reg_info)
    with open("./callback_func/mul_baseline.cc", "r", encoding="utf-8") as file:
        file_content = file.read()
        assert ''.join(file_content.split()) == ''.join(callback_func.split())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_code_generator_arg_min():
    """
    Feature: Custom op testcase
    Description: Test case for generating code of ArgMin op.
    Expectation: The generated code matches the baseline code.
    """
    reg_info = CustomRegOp("aclnnArgMin") \
        .input(0, "x", "required") \
        .attr("dim", "required", "int") \
        .attr("keep_dim", "required", "bool") \
        .output(0, "z", "required") \
        .dtype_format(DataType.F16_Default, DataType.F16_Default) \
        .target("Ascend") \
        .get_op_info()

    code_generator = CustomCodeGenerator()
    callback_func = code_generator.generate_callback_by_reg_info("aclnnArgMin", reg_info)
    with open("./callback_func/arg_min_baseline.cc", "r", encoding="utf-8") as file:
        file_content = file.read()
        assert ''.join(file_content.split()) == ''.join(callback_func.split())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_code_generator_batch_norm():
    """
    Feature: Custom op testcase
    Description: Test case for generating code of BatchNorm op.
    Expectation: The generated code matches the baseline code.
    """
    reg_info = CustomRegOp("aclnnBatchNorm") \
        .input(0, "x", "required") \
        .input(1, "scale", "required") \
        .input(2, "bias", "required") \
        .input(3, "mean", "required") \
        .input(4, "var", "required") \
        .attr("training", "required", "bool") \
        .attr("momentum", "required", "float") \
        .attr("eps", "required", "float") \
        .output(0, "output", "required") \
        .output(1, "saved_mean", "required") \
        .output(2, "saved_variance", "required") \
        .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                      DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
        .target("Ascend") \
        .get_op_info()

    code_generator = CustomCodeGenerator()
    callback_func = code_generator.generate_callback_by_reg_info("aclnnBatchNorm", reg_info)
    with open("./callback_func/batch_norm_baseline.cc", "r", encoding="utf-8") as file:
        file_content = file.read()
        assert ''.join(file_content.split()) == ''.join(callback_func.split())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='essential')
def test_code_generator_topk():
    """
    Feature: Custom op testcase
    Description: Test case for generating info of MoeSoftMaxTopk op.
    Expectation: The generated registration info matches the expected structure.
    """
    info_generator = CustomInfoGenerator("MoeSoftMaxTopk")
    aclnn_api_types = info_generator.get_aclnn_api_types()
    expected_api_types = ['aclTensor*', 'int64_t', 'aclTensor*', 'aclTensor*', 'uint64_t*', 'aclOpExecutor**']
    assert aclnn_api_types == expected_api_types

    reg_info = info_generator.generate_custom_reg_op()
    expect_reg_info = CustomRegOp("MoeSoftMaxTopk") \
        .input(0, "x", "required") \
        .attr("k", "optional", "int", "all") \
        .output(0, "y", "required") \
        .output(1, "indices", "required") \
        .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.I32_Default) \
        .target("Ascend") \
        .get_op_info()
    assert reg_info == expect_reg_info

    code_generator = CustomCodeGenerator()
    reg_info_callback_func = code_generator.generate_callback_by_reg_info("MoeSoftMaxTopk", reg_info)
    input_type_callback_func = code_generator.generate_callback_by_types("MoeSoftMaxTopk", reg_info, aclnn_api_types)
    print(reg_info_callback_func)
    assert reg_info_callback_func == input_type_callback_func
