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
Generate pyboost function from pyboost_op.yaml
"""

import os
import re
import pathlib
from dataclasses import dataclass
import pyboost_utils
from pyboost_utils import get_convert_type_str, get_input_dtype, get_return_type, tuple_input_to_cpp_type, \
    number_input_to_cpp_type, get_const_number_convert, get_tuple_input_convert, get_pyboost_name, is_cube, \
    AclnnUtils, get_disable_flag, is_optional_param, get_value_convert_type_str, is_pyboost_enable
import template
from template import CppTemplate
from op_proto import OpProto
from gen_utils import check_change_and_replace_file, py_licence_str, write_file


@dataclass
class FuncHeaderData:
    work_path: str
    op_header_template_path: list
    code_generate_path: list
    op_name_str: str
    operator_name: str
    call_args_with_type: list
    cpp_func_return: str


def generate_pyboost_base_op_header_code(work_path, op_name_str, operator_name, call_args_with_type, cpp_func_return):
    """ generate_pyboost_base_op_header_code """
    pyboost_op_header_str = template.PYBOOST_BASE_OP_DEFINE_TEMPLATE.replace(op_name=op_name_str,
                                                                             op_name_upper=op_name_str.upper(),
                                                                             call_args=call_args_with_type,
                                                                             return_type=cpp_func_return)
    op_header_dir_path = os.path.join(work_path, "mindspore/ccsrc/kernel/pyboost/auto_generate/")
    pathlib.Path(op_header_dir_path).mkdir(parents=True, exist_ok=True)
    tmp_op_file_path = os.path.join(op_header_dir_path, "tmp_" + operator_name + ".h")
    dst_op_file_path = os.path.join(op_header_dir_path, operator_name + ".h")
    write_file(tmp_op_file_path, pyboost_op_header_str)
    check_change_and_replace_file(dst_op_file_path, tmp_op_file_path)


def generate_pyboost_op_header_code(header_data: FuncHeaderData):
    """ generate_pyboost_op_header_code """

    for tpl_path, gen_path in zip(header_data.op_header_template_path, header_data.code_generate_path):
        pyboost_op_str = tpl_path.replace(op_name=header_data.op_name_str,
                                          op_name_upper=header_data.op_name_str.upper(),
                                          operator_name=header_data.operator_name,
                                          call_args_with_type=header_data.call_args_with_type,
                                          return_type=header_data.cpp_func_return)
        op_header_dir_path = os.path.join(header_data.work_path, gen_path)
        pathlib.Path(op_header_dir_path).mkdir(parents=True, exist_ok=True)
        tmp_op_file_path = os.path.join(op_header_dir_path, "tmp_" + header_data.operator_name + ".h")
        dst_op_file_path = os.path.join(op_header_dir_path, header_data.operator_name + ".h")
        write_file(tmp_op_file_path, pyboost_op_str)
        check_change_and_replace_file(dst_op_file_path, tmp_op_file_path)


class TemplatePaths:
    """
    template paths for code auto generation
    """

    def __init__(self, op_header_template_path, op_call_template_path, op_source_template_path, op_custom_template_path,
                 op_view_template_path, code_generate_path):
        self.op_header_template_path = op_header_template_path
        self.op_call_template_path = op_call_template_path
        self.op_source_template_path = op_source_template_path
        self.op_custom_template_path = op_custom_template_path
        self.op_view_template_path = op_view_template_path
        self.code_generate_path = code_generate_path


def generate_malloc_input(need_malloc_tensors):
    """
    generate malloc inputs
    :param need_malloc_tensors:
    :return:
    """
    malloc_inputs = ''
    args_list = ''
    for item in need_malloc_tensors:
        args_list += f'{item}, '
    args_list = args_list[:-2]
    if args_list:
        malloc_inputs += f'PyBoostUtils::MallocOpInputs(device_context, {args_list});\n'
    return malloc_inputs


def generate_get_inputs_kernel_tensors(call_args):
    """
    generate get inputs kernel tensors
    :param call_args:
    :return:
    """
    inputs_kernel_tensors = ''
    args_list = ''
    for item in call_args:
        args_list += f'{item}, '
    args_list = args_list[:-2]
    if args_list:
        inputs_kernel_tensors += f'const auto &input_address_info = PyBoostUtils::GetAddressInfo(' \
                                 f'device_context, op->stream_id(), op->input_abs(), {args_list});\n'
    return inputs_kernel_tensors


def generate_create_input_address(need_malloc_tensors):
    """create input address"""
    create_input_address = ''
    args_list = ''
    for item in need_malloc_tensors:
        args_list += f'{item}, '
    args_list = args_list[:-2]
    if args_list:
        create_input_address = f'PyBoostUtils::PrepareOpInputs(device_context_, op->stream_id(), {args_list});\n'
    return create_input_address


def generate_tensor_cpu_cast_input_code(call_args_with_tensor, call_tensors):
    """ generate_tensor_cpu_cast_input_code """
    cast_input = ""
    real_call_args_tensor = call_args_with_tensor.copy()
    for i, tensor in enumerate(call_args_with_tensor):
        is_tuple_tensor = real_call_args_tensor[i].endswith("_vector")
        is_tensor = real_call_args_tensor[i] in call_tensors
        if is_tensor:
            cast_input += f'const auto &real_{tensor} = PyBoostUtils::CastTensor({tensor}, ' \
                          f'select_kernel.input_type()[{i}].dtype, "CPU");\n'
            real_call_args_tensor[i] = "real_" + real_call_args_tensor[i]
        if is_tuple_tensor:
            cast_input += f'const auto &real_{tensor} = PyBoostUtils::CastTensor({tensor}, ' \
                          f'select_kernel.input_type()[{i}].dtype, "CPU");\n'
            real_call_args_tensor[i] = "PyBoostUtils::ConvertTensorVectorToTuple(real_" + real_call_args_tensor[i] + ")"
    if cast_input != "":
        cast_input = "auto &select_kernel = kernel_attr_pair.second;\n" + cast_input
    return cast_input, real_call_args_tensor


def generate_pyboost_op_source_code(work_path, op_proto, template_paths, converter):
    """ generate_pyboost_op_source_code """
    # PyBoost source generate
    operator_name = converter.functional_name
    call_args_tensor = []
    for type, arg_name in zip(converter.call_args_types, converter.call_args):
        if type in ("BaseTensorPtr", "std::optional<BaseTensorPtr>"):
            call_args_tensor.append(arg_name)

    for call_tpl, src_tpl, view_tpl, cus_tpl, gen_path in zip(template_paths.op_call_template_path,
                                                              template_paths.op_source_template_path,
                                                              template_paths.op_view_template_path,
                                                              template_paths.op_custom_template_path,
                                                              template_paths.code_generate_path):
        is_ascend = 'ascend' in gen_path
        is_cpu = 'cpu' in gen_path
        is_gpu = 'gpu' in gen_path
        malloc_inputs = generate_malloc_input(converter.need_malloc_tensors)
        create_input_address = generate_create_input_address(converter.need_malloc_tensors)
        get_inputs_kernel_tensors = generate_get_inputs_kernel_tensors(converter.call_args_with_tensor)

        # call_impl
        call_impl = ''
        customize_include = ''
        op_name_str = op_proto.class_name
        cube_math_type = ''
        get_cube_math_type = ''
        real_output = ', ' + converter.op_outputs
        proto_operator_name = op_proto.operator_name
        register_custom_kernel = ''
        if is_ascend and op_proto.ascend != 'default':
            call_impl = cus_tpl.replace(call_args=converter.call_args,
                                        return_values=converter.call_func_outputs,
                                        customize_func=op_proto.ascend + "Customize",
                                        )
            customize_include = "#include \"plugin/device/ascend/kernel/pyboost/customize/{}.h\"".format(
                operator_name.lower())
        elif is_cpu and op_proto.cpu != 'default':
            call_impl = cus_tpl.replace(call_args=converter.call_args,
                                        return_values=converter.call_func_outputs,
                                        customize_func=op_proto.cpu + "Customize",
                                        )
            customize_include = "#include \"plugin/device/cpu/kernel/pyboost/customize/{}.h\"".format(
                operator_name.lower())
            register_custom_kernel = "MS_REG_PYBOOST_CPU_CUSTOM_KERNEL({});".format(op_name_str)
        elif is_gpu and op_proto.gpu != 'default':
            call_impl = cus_tpl.replace(call_args=converter.call_args,
                                        return_values=converter.call_func_outputs,
                                        customize_func=op_proto.gpu + "Customize",
                                        )
            customize_include = "#include \"plugin/device/gpu/kernel/pyboost/customize/{}.h\"".format(
                operator_name.lower())
            register_custom_kernel = "MS_REG_PYBOOST_GPU_CUSTOM_KERNEL({});".format(op_name_str)
        elif op_proto.is_view:
            set_output_abs = "SetOutputAbstract();"
            if converter.call_func_outputs == "outputs_":
                set_output_abs = "SetOutputTupleAbstract();"
            call_impl = view_tpl.replace(op_name=op_proto.class_name,
                                         call_args=converter.call_args,
                                         call_tensors=call_args_tensor,
                                         return_values=converter.call_func_outputs,
                                         input=converter.call_args[0],
                                         set_output_abs=set_output_abs)
            customize_include = "#include \"mindspore/core/ops/view/{}_strides_calc.h\"".format(proto_operator_name)
        else:
            cast_input_code, real_call_args_tensor = generate_tensor_cpu_cast_input_code(
                converter.call_args_with_tensor, call_args_tensor)
            if is_ascend and is_cube(op_proto.class_name):
                get_cube_math_type = f'// cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION\n'
                get_cube_math_type += "auto cube_math_type = GetCubeMathType();"
                cube_math_type = ', cube_math_type'
            aclnn_name = AclnnUtils.get_aclnn_interface(op_name_str)
            if converter.inplace_process != '':
                real_output = ''
            customize_include = '#include "ops/auto_generate/gen_ops_primitive.h"'

            call_impl = call_tpl.replace(aclnn_name=aclnn_name,
                                         call_args=converter.call_args,
                                         call_tensors=call_args_tensor,
                                         value_tuple_convert=converter.value_tuple_convert,
                                         const_number_convert=converter.const_number_convert,
                                         create_input_address=create_input_address,
                                         tensor_list_convert=converter.tensor_list_convert,
                                         call_args_with_tensor=converter.call_args_with_tensor,
                                         malloc_inputs=malloc_inputs,
                                         get_inputs_kernel_tensors=get_inputs_kernel_tensors,
                                         get_cube_math_type=get_cube_math_type,
                                         cube_math_type=cube_math_type,
                                         real_call_args=converter.call_args_after_convert,
                                         return_values=converter.call_func_outputs,
                                         outputs=real_output,
                                         inplace_process=converter.inplace_process,
                                         cast_input_code=cast_input_code,
                                         real_call_args_tensor=real_call_args_tensor,
                                         class_name=op_proto.class_name,
                                         op_name_str=op_name_str)

        pyboost_op_source_str = src_tpl.replace(op_name=op_name_str,
                                                operator_name=operator_name,
                                                call_args_with_type=converter.call_args_with_types,
                                                return_type=converter.cpp_func_return,
                                                customize_include=customize_include,
                                                call_impl=call_impl,
                                                register_custom_kernel=register_custom_kernel)
        op_header_dir_path = os.path.join(work_path, gen_path)
        tmp_op_source_file_path = os.path.join(op_header_dir_path, "tmp_" + operator_name.lower() + ".cc")
        dst_op_source_file_path = os.path.join(op_header_dir_path, operator_name.lower() + ".cc")
        write_file(tmp_op_source_file_path, pyboost_op_source_str)
        check_change_and_replace_file(dst_op_source_file_path, tmp_op_source_file_path)


def generate_pyboost_op_register_source_code(work_path, all_ops, all_operator_names):
    """ generate_pyboost_op_register_source_code """
    include_str = ''
    factory_str = ''
    for op_name in all_ops:
        factory_str += "template class OpFactory<{0}>;\n".format(op_name)
    for operator_name in all_operator_names:
        include_str += "#include \"kernel/pyboost/auto_generate/{0}.h\"\n".format(operator_name)
    op_register_file_str = template.PYBOOST_OP_REGISTER_TEMPLATE.replace(op_includes=include_str,
                                                                         op_factory_templates=factory_str)
    op_register_dir_path = os.path.join(work_path, "mindspore/ccsrc/kernel/pyboost/auto_generate/")
    pathlib.Path(op_register_dir_path).mkdir(parents=True, exist_ok=True)
    tmp_op_register_file_path = os.path.join(op_register_dir_path, "tmp_" + "op_register.cc")
    dst_op_register_file_path = os.path.join(op_register_dir_path, "op_register.cc")
    write_file(tmp_op_register_file_path, op_register_file_str)
    check_change_and_replace_file(dst_op_register_file_path, tmp_op_register_file_path)


def generate_pyboost_op_return_code(op_proto):
    """ generate_pyboost_op_return_code """
    returns_type = []
    for return_obj in op_proto.returns:
        returns_type.append(get_return_type(return_obj.arg_dtype))
    if len(returns_type) == 1:
        cpp_func_return = returns_type[0]
    elif not returns_type:
        raise Exception("No return")
    else:
        cpp_func_return = "std::tuple("
        cpp_func_return += ','.join(s for s in returns_type)
        cpp_func_return += ")"
    return returns_type, cpp_func_return


def generate_pyboost_op_func_return_type(op_proto):
    """ generate_pyboost_op_func_return_type """
    returns_type = []
    type_convert_to_base = {
        'std::vector<tensor::TensorPtr>': 'std::vector<tensor::BaseTensorPtr>',
        'tensor::TensorPtr': 'tensor::BaseTensorPtr'
    }
    for return_obj in op_proto.returns:
        temp_return = get_return_type(return_obj.arg_dtype)
        if temp_return in type_convert_to_base:
            returns_type.append(type_convert_to_base[temp_return])
        else:
            raise Exception("Not return found")
    if len(returns_type) == 1:
        cpp_func_return = returns_type[0]
    elif len(returns_type) > 1:
        cpp_func_return = "std::tuple<"
        cpp_func_return += ','.join(s for s in returns_type)
        cpp_func_return += ">"
    else:
        raise Exception("Not return found")
    return cpp_func_return


def generate_pyboost_outputs(op_proto):
    """ generate_pyboost_outputs """
    op_outputs = ''
    call_outputs = ''
    returns_type = []
    for return_obj in op_proto.returns:
        returns_type.append(get_return_type(return_obj.arg_dtype))

    if len(returns_type) == 1:
        if returns_type[0] == 'tensor::TensorPtr':
            op_outputs = 'outputs[0]'
            call_outputs = 'outputs_[0]'
        elif returns_type[0] == "std::vector<tensor::TensorPtr>":
            op_outputs = 'outputs'
            call_outputs = 'outputs_'
        else:
            raise Exception("Not support return type {}".format(returns_type[0]))
    elif len(returns_type) > 1:
        outputs_str = ''
        for i in range(len(returns_type)):
            outputs_str += 'outputs[{}],'.format(i)
        op_outputs = outputs_str[:-1]

        outputs_str = ''
        for i in range(len(returns_type)):
            outputs_str += 'outputs_[{}],'.format(i)
        outputs_str = outputs_str[:-1]
        call_outputs = "std::make_tuple(" + outputs_str + ")"

    return op_outputs, call_outputs


def generate_ops_header_files(work_path, yaml_data):
    """
    :param work_path:
    :param yaml_data:
    :return: void
    """
    extern_str = ''
    extern_template = CppTemplate("MS_EXPORT extern OpDef g${op_name};\n")
    for operator_name, operator_data in yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        extern_str += extern_template.replace(op_name=op_proto.class_name)
    ops_header_file = template.GEN_OPS_DEF_HEADER_TEMPLATE.replace(extern_variable=extern_str)
    dir_path = os.path.join(work_path, "mindspore/core/ops/auto_generate")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    dst_file_path = os.path.join(dir_path, "gen_ops_def.h")
    tmp_file_path = os.path.join(dir_path, "tmp_gen_ops_def.h")
    write_file(tmp_file_path, ops_header_file)
    check_change_and_replace_file(dst_file_path, tmp_file_path)


def generate_parser_func(op_proto: OpProto) -> str:
    """
    Generate parser func
    :param op_proto:
    :return: str
    """
    convert_template = CppTemplate("auto $arg_name = converter.${convert_func}(args, $arg_index);\n")
    parser_func_str = ''
    for index, arg in enumerate(op_proto.op_args):
        is_optional = is_optional_param(arg)
        if arg.is_type_id:
            arg.arg_dtype = 'type'
        convert_type_str = get_convert_type_str(arg.arg_dtype, is_optional)
        parser_func_str += convert_template.replace(arg_name=arg.arg_name, convert_func=convert_type_str,
                                                    arg_index=pyboost_utils.get_index(index))
    return parser_func_str


def get_convert_tensor_template():
    """
    Get convert tensor template
    """
    convert_to_tensor_template = CppTemplate(
        'auto ${output} = PyNativeAlgo::Common::ConvertStubNodeToTensor(${input}, ${need_contiguous}, '\
        'op_run_info->requires_grad);\n')
    convert_to_tensor_list_template = CppTemplate(
        'auto ${output} = PyNativeAlgo::Common::ConvertStubNodeToValueTuple(${input}, ${need_contiguous}, '\
        'op_run_info->requires_grad);\n')
    return convert_to_tensor_template, convert_to_tensor_list_template


def generate_pyboost_functions(work_path, yaml_data):
    """
    Generate pyboost functions file from yaml.
    """
    pyboost_func_str = ''
    pyboost_func_pybind_def = ''
    pyboost_func_include_headers_str = ''
    pyboost_func_include_header_template = CppTemplate("#include \"kernel/pyboost/auto_generate/${operator_name}.h\"\n")
    for operator_name, operator_data in yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        if not op_proto.is_dispatch:
            continue
        op_def_name_str = f"g{op_proto.class_name}"
        operator_name = op_proto.operator_name
        op_name_str = op_proto.class_name
        op_args_str = [op_arg.arg_name for op_arg in op_proto.op_args]
        parser_body_str = generate_parser_func(op_proto)
        convert_to_tensor_template, convert_to_tensor_list_template = get_convert_tensor_template()

        grad_args_str = []
        call_args_str = []
        cast_args_str = []
        convert_stub_str = ''
        optional_to_value_str = ''
        need_contiguous = 'true'
        value_str = '_value'
        if op_proto.is_view:
            # view/aclnn op no need to contiguous tensor.
            need_contiguous = 'false'
        for op_arg in op_proto.op_args:
            cast_str = 'cast_'
            convert_optional_to_value_template = CppTemplate(
                "auto ${output} = PyNativeAlgo::PyBoost::OptionalToValue(${input});\n")
            if pyboost_utils.is_tensor(op_arg):
                if is_optional_param(op_arg):
                    convert_stub_output_name = op_arg.arg_name + '_optional'
                    convert_stub_str += convert_to_tensor_template.replace(output=convert_stub_output_name,
                                                                           input=op_arg.arg_name,
                                                                           need_contiguous=need_contiguous)
                    cast_output = cast_str + convert_stub_output_name

                    convert_optional_to_value_name = op_arg.arg_name + value_str
                    optional_to_value_str += \
                        convert_optional_to_value_template.replace(input=cast_output,
                                                                   output=convert_optional_to_value_name)
                    call_arg = convert_stub_output_name
                    grad_arg = convert_optional_to_value_name
                    cast_arg = cast_output
                else:
                    convert_stub_output_name = op_arg.arg_name + "_tensor"
                    convert_stub_str += convert_to_tensor_template.replace(input=op_arg.arg_name,
                                                                           output=convert_stub_output_name,
                                                                           need_contiguous=need_contiguous)
                    call_arg = convert_stub_output_name
                    grad_arg = cast_str + convert_stub_output_name
                    cast_arg = grad_arg
            elif pyboost_utils.is_tensor_list(op_arg):
                if is_optional_param(op_arg):
                    # to adapt the cases that TensorList is optional.
                    convert_stub_output_name = op_arg.arg_name + '_optional'
                    convert_stub_str += convert_to_tensor_list_template.replace(output=convert_stub_output_name,
                                                                                input=op_arg.arg_name,
                                                                                need_contiguous=need_contiguous)
                    cast_output = cast_str + convert_stub_output_name

                    convert_optional_to_value_name = op_arg.arg_name + value_str
                    optional_to_value_str += \
                        convert_optional_to_value_template.replace(input=cast_output,
                                                                   output=convert_optional_to_value_name)
                    call_arg = convert_stub_output_name
                    grad_arg = convert_optional_to_value_name
                    cast_arg = cast_output
                else:
                    convert_stub_output_name = op_arg.arg_name + "_tensor_list"
                    convert_stub_str += convert_to_tensor_list_template.replace(input=op_arg.arg_name,
                                                                                output=convert_stub_output_name,
                                                                                need_contiguous=need_contiguous)
                    call_arg = convert_stub_output_name
                    grad_arg = cast_str + convert_stub_output_name
                    cast_arg = grad_arg
            else:
                call_arg = op_arg.arg_name
                grad_arg = cast_str + op_arg.arg_name
                cast_arg = grad_arg
                if is_optional_param(op_arg):
                    convert_optional_to_value_name = op_arg.arg_name + value_str
                    optional_to_value_str += \
                        convert_optional_to_value_template.replace(input=call_arg,
                                                                   output=convert_optional_to_value_name)
                    grad_arg = convert_optional_to_value_name
            grad_args_str.append(grad_arg)
            call_args_str.append(call_arg)
            cast_args_str.append(cast_arg)
        type_num, same_type = gen_signature_same_type_table(op_proto.indexes, operator_data)
        pyboost_func_str += template.PYBOOST_FUNCTION_TEMPLATE.replace(func_name=op_proto.pyboost_function_name,
                                                                       op_def_name=op_def_name_str, same_type=same_type,
                                                                       type_num=type_num, parser_body=parser_body_str,
                                                                       op_name=op_name_str,
                                                                       convert_stub=convert_stub_str,
                                                                       optional_to_value=optional_to_value_str,
                                                                       call_args=call_args_str, grad_args=grad_args_str,
                                                                       cast_args=cast_args_str, op_args=op_args_str,
                                                                       class_name=op_proto.class_name)
        pyboost_func_str = pyboost_func_str + template.NEW_LINE + template.NEW_LINE
        pyboost_func_pybind_def += template.REGISTER_DEFINE_TEMPLATE.replace(
            pyboost_op_name=get_pyboost_name(op_proto.operator_name),
            pyboost_cfunc_name=op_proto.pyboost_function_name, class_name=op_proto.class_name)
        pyboost_func_include_headers_str += pyboost_func_include_header_template.replace(operator_name=operator_name)
    register_func_str = template.REGISTER_TEMPLATE.replace(register_func=pyboost_func_pybind_def)
    pyboost_func_file = template.PYBOOST_HEADER_TEMPLATE.replace(include_op_header=pyboost_func_include_headers_str,
                                                                 function_body=pyboost_func_str,
                                                                 register_function_body=register_func_str)
    dir_path = os.path.join(work_path, "mindspore/ccsrc/pipeline/pynative/op_function/auto_generate")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    tmp_file_path = os.path.join(dir_path, "tmp_pyboost_functions.cc")
    dst_file_path = os.path.join(dir_path, "pyboost_functions.cc")
    write_file(tmp_file_path, pyboost_func_file)
    check_change_and_replace_file(dst_file_path, tmp_file_path)


def convert_value_type(op_proto: OpProto) -> str:
    """
    Generate parser func
    :param op_proto:
    :return: str
    """
    convert_template = CppTemplate(
        "auto $arg_name = ValueConverter::${convert_func}(op_runner_info->inputs, $arg_index);\n")
    parser_func_str = ''
    for index, arg in enumerate(op_proto.op_args):
        is_optional = is_optional_param(arg)
        convert_type_str = get_value_convert_type_str(arg.arg_dtype, is_optional)
        parser_func_str += convert_template.replace(arg_name=arg.arg_name, convert_func=convert_type_str,
                                                    arg_index=pyboost_utils.get_index(index))
    return parser_func_str


def contiguous_tensor_value(op_proto: OpProto) -> str:
    """
    Generate parser func
    :param op_proto:
    :return: str
    """
    # Do nothing in view op
    if op_proto.is_view:
        return ''
    contiguous_template = CppTemplate(
        "$arg_name = ValueConverter::ContiguousTensorValue(op_runner_info, $arg_name);\n")
    contiguous_func_str = ''
    need_contiguous_dtype = {'tensor', 'tuple[tensor]'}
    for arg in op_proto.op_args:
        if arg.arg_dtype not in need_contiguous_dtype:
            continue
        contiguous_func_str += contiguous_template.replace(arg_name=arg.arg_name)
    return contiguous_func_str


def generate_pyboost_grad_functions(work_path, yaml_data):
    """
    Generate pyboostgrad  functions file from yaml.
    """
    pyboost_func_str = ''
    pyboost_func_reg_def = ''
    pyboost_func_include_headers_str = ''
    pyboost_func_include_header_template = CppTemplate("#include \"kernel/pyboost/auto_generate/${operator_name}.h\"\n")
    for operator_name, operator_data in yaml_data.items():
        if not is_pyboost_enable(operator_data):
            continue
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        if not op_proto.is_dispatch:
            continue
        operator_name = op_proto.operator_name
        op_name_str = op_proto.class_name
        op_args_str = [op_arg.arg_name for op_arg in op_proto.op_args]
        convert_value_type_str = convert_value_type(op_proto)
        convert_value_type_str += contiguous_tensor_value(op_proto)

        call_args_str = []
        for op_arg in op_proto.op_args:
            call_arg = op_arg.arg_name
            call_args_str.append(call_arg)
        pyboost_func_str += template.PYBOOST_GRAD_FUNCTION_TEMPLATE.replace(func_name=op_proto.pyboost_function_name,
                                                                            op_name=op_name_str,
                                                                            op_args=op_args_str,
                                                                            convert_body=convert_value_type_str,
                                                                            call_args=call_args_str)
        pyboost_func_str = pyboost_func_str + template.NEW_LINE
        pyboost_func_reg_def += template.REGISTER_PYBOOST_GRAD_DEFINE_TEMPLATE.replace(
            pyboost_op_name=op_proto.class_name,
            pyboost_cfunc_name=op_proto.pyboost_function_name)
        pyboost_func_include_headers_str += pyboost_func_include_header_template.replace(operator_name=operator_name)

    register_func_str = template.REGISTER_PYBOOST_GRAD_TEMPLATE.replace(register_func=pyboost_func_reg_def)
    pyboost_func_file = \
        template.PYBOOST_GRAD_HEADER_TEMPLATE.replace(include_op_header=pyboost_func_include_headers_str,
                                                      function_body=pyboost_func_str,
                                                      register_function_body=register_func_str)
    dir_path = os.path.join(work_path, "mindspore/ccsrc/runtime/pynative/op_function/auto_generate")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    tmp_file_path = os.path.join(dir_path, "tmp_pyboost_grad_functions.cc")
    dst_file_path = os.path.join(dir_path, "pyboost_grad_functions.cc")
    write_file(tmp_file_path, pyboost_func_file)
    check_change_and_replace_file(dst_file_path, tmp_file_path)


def generate_inplace_process_cpp_code(op_proto):
    """ generate_ref_process_cpp_code """
    inplace_process = f'// RefOps update output by input tensor\n'
    has_ref = False
    for index, return_obj in enumerate(op_proto.returns):
        if return_obj.inplace != '':
            inplace_process += f'outputs_[{index}]->set_device_address(' \
                               f'{return_obj.inplace}_tensor->device_address()); '
            has_ref = True
            break
    if has_ref:
        return inplace_process
    return ''


def get_auto_generate_template():
    """
    get template collections
    :return: TemplatePaths
    """
    op_header_template_path = [template.PYBOOST_ASCEND_OP_HEADER_TEMPLATE, template.PYBOOST_GPU_OP_HEADER_TEMPLATE,
                               template.PYBOOST_CPU_OP_HEADER_TEMPLATE]
    op_call_template_path = [template.PYBOOST_ASCEND_CALL_TEMPLATE, template.PYBOOST_GPU_CALL_TEMPLATE,
                             template.PYBOOST_CPU_CALL_TEMPLATE]
    op_source_template_path = [template.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE, template.PYBOOST_GPU_OP_SOURCE_TEMPLATE,
                               template.PYBOOST_CPU_OP_SOURCE_TEMPLATE]
    op_custom_template_path = [template.PYBOOST_ASCEND_CUSTOMIZE_CALL_TEMPLATE,
                               template.PYBOOST_GPU_CUSTOMIZE_CALL_TEMPLATE,
                               template.PYBOOST_CPU_CUSTOMIZE_CALL_TEMPLATE]
    op_view_template_path = [template.PYBOOST_ASCEND_VIEW_CALL_TEMPLATE, template.PYBOOST_GPU_VIEW_CALL_TEMPLATE,
                             template.PYBOOST_CPU_VIEW_CALL_TEMPLATE]
    code_generate_path = ["mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/auto_generate/",
                          "mindspore/ccsrc/plugin/device/gpu/kernel/pyboost/auto_generate/",
                          "mindspore/ccsrc/plugin/device/cpu/kernel/pyboost/auto_generate/"]
    return TemplatePaths(op_header_template_path, op_call_template_path, op_source_template_path,
                         op_custom_template_path,
                         op_view_template_path, code_generate_path)


class OpTemplateConverter:
    """
    template converter
    """

    def __init__(self, op_proto):
        self.op_proto = op_proto
        self.op_name = op_proto.class_name
        self.functional_name = op_proto.operator_name
        self.call_args = self.parse_original_call_args(op_proto.op_args)
        self.call_args_types = self.parse_call_args_types(op_proto.op_args)
        self.call_args_with_types = self.parse_call_args_with_types(self.call_args, self.call_args_types)
        self.need_malloc_tensors, self.tensor_list_convert, self.call_args_with_tensor = \
            self.parse_need_malloc_tensors(op_proto.op_args, self.call_args)
        self.call_args_after_convert, self.value_tuple_convert, self.const_number_convert = \
            self.op_args_converter(op_proto.op_args, self.call_args)
        self.cpp_func_return = generate_pyboost_op_func_return_type(op_proto)
        self.op_outputs, self.call_func_outputs = generate_pyboost_outputs(op_proto)
        self.inplace_process = generate_inplace_process_cpp_code(op_proto)

    @staticmethod
    def parse_call_args_types(op_args):
        """
        :param op_args:
        :return: call_args_types
        """
        call_args_types = []
        for op_arg in op_args:
            is_optional = is_optional_param(op_arg)
            call_args_types.append(get_input_dtype(op_arg.arg_dtype, is_optional))
        return call_args_types

    @staticmethod
    def parse_call_args_with_types(call_args, call_args_types):
        """
        :param call_args:
        :param call_args_types:
        :return: call_args_with_types
        """
        call_args_with_types = []
        for type_name, arg_name in zip(call_args_types, call_args):
            call_args_with_types.append("const " + type_name + " &" + arg_name)
        return call_args_with_types


    @staticmethod
    def parse_need_malloc_tensors(op_args, call_args):
        """
        :param op_args:
        :param call_args:
        :return: need_malloc_tensors
        """
        need_malloc_tensors = []
        tensor_list_convert = []
        call_args_with_tensor = []
        for op_arg, call_arg in zip(op_args, call_args):
            if pyboost_utils.is_tensor(op_arg):
                call_arg = op_arg.arg_name + "_tensor"
                need_malloc_tensors.append(call_arg)
                call_args_with_tensor.append(call_arg)
            elif tuple_input_to_cpp_type(op_arg.arg_dtype) and pyboost_utils.is_tensor_list(op_arg):
                need_malloc_tensors.append(call_arg + "_vector")
                tensor_list_convert.append(get_tuple_input_convert(call_arg, op_arg.arg_dtype))
                call_args_with_tensor.append(call_arg + "_vector")
            else:
                call_args_with_tensor.append(call_arg)
        return need_malloc_tensors, tensor_list_convert, call_args_with_tensor


    @staticmethod
    def parse_original_call_args(op_args):
        """
        :param op_args:
        :return: call_args
        """
        call_args = []
        for op_arg in op_args:
            if pyboost_utils.is_tensor(op_arg):
                call_arg = op_arg.arg_name + "_tensor"
            elif pyboost_utils.is_tensor_list(op_arg):
                call_arg = op_arg.arg_name + "_tensor_list"
            else:
                call_arg = op_arg.arg_name
            call_args.append(call_arg)
        return call_args

    @staticmethod
    def op_args_converter(op_args, call_args):
        """Convert ValutePtr to cpp data type"""
        call_args_after_convert = []
        value_tuple_convert = []
        const_number_convert = []
        for op_arg, call_arg in zip(op_args, call_args):
            if number_input_to_cpp_type(op_arg.arg_dtype):
                call_args_after_convert.append(call_arg + "_imm")
                const_number_convert.append(get_const_number_convert(call_arg, op_arg))
            elif tuple_input_to_cpp_type(op_arg.arg_dtype):
                call_args_after_convert.append(call_arg + "_vector")
                value_tuple_convert.append(get_tuple_input_convert(call_arg, op_arg.arg_dtype))
            else:
                call_args_after_convert.append(call_arg)
        if const_number_convert:
            const_number_convert.insert(0, '// Convert ValuePtr to c++ scalar\n')
        if value_tuple_convert:
            value_tuple_convert.insert(0, '// ValueTuple to std::vector\n')
        return call_args_after_convert, value_tuple_convert, const_number_convert


def delete_residual_files(work_path, all_operator_name, code_generate_path_list):
    """
    Delete residual files.
    """
    code_generate_path_list.append("mindspore/ccsrc/kernel/pyboost/auto_generate/")
    for code_generate_path in code_generate_path_list:
        all_files_name = []
        code_generate_path = os.path.join(work_path, code_generate_path)
        if os.path.exists(code_generate_path):
            all_files_name = os.listdir(code_generate_path)
        all_registered_op = set(item.split(".")[0] for item in all_files_name)
        need_clean_op = all_registered_op - set(all_operator_name)
        for file in all_files_name:
            if file == "op_register.cc":
                continue
            for clean_name in need_clean_op:
                judge_file = file.split(".")[0]
                if judge_file == clean_name:
                    file_path = os.path.join(code_generate_path, file)
                    if os.path.exists(file_path):
                        os.remove(file_path)


def generate_pyboost_op_cpp_code(work_path, yaml_data):
    """
    Generate pyboost op cpp code from yaml.
    """

    all_op_names = []
    all_functional_names = []
    all_operator_name = []
    for operator_name, operator_data in yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        if not op_proto.is_dispatch:
            continue
        template_paths = get_auto_generate_template()
        converter = OpTemplateConverter(op_proto)
        functional_name = converter.functional_name

        op_name_str = converter.op_name

        all_op_names.append(op_name_str)
        all_operator_name.append(operator_name)
        all_functional_names.append(functional_name)

        call_args_with_types = converter.call_args_with_types
        cpp_func_return = converter.cpp_func_return

        generate_pyboost_base_op_header_code(work_path, op_name_str, functional_name, call_args_with_types,
                                             cpp_func_return)
        header_data = FuncHeaderData(work_path, template_paths.op_header_template_path,
                                     template_paths.code_generate_path, op_name_str,
                                     functional_name, call_args_with_types, cpp_func_return)
        generate_pyboost_op_header_code(header_data)
        generate_pyboost_op_source_code(work_path, op_proto, template_paths, converter)
    delete_residual_files(work_path, all_operator_name, template_paths.code_generate_path)
    generate_pyboost_op_register_source_code(work_path, all_op_names, all_functional_names)


def gen_pyboost_inner_prim(work_path, op_yaml_data):
    """
    gen pyboost inner prim
    :param work_path:
    :param op_yaml_data:
    :return:
    """
    gen_py = ''
    gen_header = py_licence_str + template.IMPORT_PYBOOST_PRIM_HEADER
    for operator_name, operator_data in op_yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        if not op_proto.is_pyboost:
            continue
        if not op_proto.prim_init:
            continue
        gen_header += template.PYBOOST_PY_FUNC_IMPORT_HEADEAR.replace(class_name=op_proto.class_name)
        args = operator_data.get('args')
        input_args = []
        processed_args = []
        process_func = ''
        for arg_name, arg_info in args.items():
            arg_handler = arg_info.get('arg_handler')
            processed_arg = arg_name
            if arg_handler is not None and arg_handler != 'dtype_to_type_id':
                process_func += f"""converted_{arg_name} = {arg_handler}('{operator_name}', '{arg_name}', {arg_name})\n"""
                processed_arg = 'converted_' + arg_name
            input_args.append(arg_name)
            processed_args.append(processed_arg)
        gen_py += template.PYTHON_PRIM_TEMPLATE.replace(class_name=op_proto.class_name, input_args=input_args,
                                                        process_func=process_func, func_impl_name=operator_name,
                                                        processed_args=processed_args)
    dir_path = os.path.join(work_path, "mindspore/python/mindspore/ops/auto_generate")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    dst_file_path = os.path.join(dir_path, "pyboost_inner_prim.py")
    tmp_file_path = os.path.join(dir_path, "tmp_pyboost_inner_prim.py")
    write_file(tmp_file_path, gen_header + gen_py)
    check_change_and_replace_file(dst_file_path, tmp_file_path)


def process_args(args):
    """
    process args
    :return: func args, input_args
    """
    func_args = []
    input_args = []
    for arg_name, arg_info in args.items():
        init_value = arg_info.get('init')
        arg_handler = arg_info.get('arg_handler')
        input_arg = arg_name
        if arg_handler is not None and arg_handler != 'dtype_to_type_id':
            input_arg = 'converted_' + arg_name
        if init_value is None:
            default_key = 'default'
            default_value = arg_info.get(default_key)
            default_value = '=' + str(default_value) if default_key in arg_info else ''
            func_args.append(arg_name + default_value)
            input_args.append(input_arg)
        else:
            if init_value == 'NO_VALUE':
                func_args.append(f"""{arg_name}""")
            else:
                func_args.append(f"""{arg_name}={init_value}""")
    return func_args, input_args


def gen_pyboost_py_func(work_path, op_yaml_data, doc_data):
    """ gen_pyboost_py_func """
    gen_py = ''
    op_desc_dict = {}

    py_header = py_licence_str + template.IMPORT_PYBOOST_FUNC_HEADER
    for operator_name, operator_desc in doc_data.items():
        desc = operator_desc.get("description")
        op_desc_dict[operator_name] = desc
    for operator_name, operator_data in op_yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        if not op_proto.is_pyboost:
            continue
        func_def = operator_data.get('function')
        func_name = operator_name
        if func_def is not None:
            func_disable = get_disable_flag(func_def)
            if func_disable:
                continue
            item = func_def.get("name")
            if item is not None:
                func_name = item
        if func_name.endswith("_ext"):
            func_name = func_name[:-4]
        else:
            continue
        func_impl_name = func_name
        if func_name.endswith("_"):
            func_impl_name = func_name[:-1]
        description = op_desc_dict.get(operator_name)
        args = operator_data.get('args')
        func_args, input_args = process_args(args)
        gen_py += template.PYBOOST_PY_FUNC_TEMPLATE.replace(func_name=func_name, description=description,
                                                            func_args=func_args,
                                                            func_impl_name=func_impl_name,
                                                            input_args=input_args)
    dir_path = os.path.join(work_path, "mindspore/python/mindspore/ops/auto_generate")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    dst_file_path = os.path.join(dir_path, "gen_extend_func.py")
    tmp_file_path = os.path.join(dir_path, "tmp_gen_extend_func.py")
    write_file(tmp_file_path, py_header + gen_py)
    check_change_and_replace_file(dst_file_path, tmp_file_path)


def gen_signature_same_type_table(args_map, operator_data):
    """
    gen signature same type table
    :param operator_name:
    :param operator_data:
    :return:
    """
    args_signature = operator_data.get('args_signature')
    signature_table = ''
    type_num = 0
    if args_signature is not None:
        dtype_group = args_signature.get('dtype_group')
        if dtype_group is not None:
            match = re.findall(r'\((.*?)\)', dtype_group)
            for item in match:
                name_args = item.replace(' ', '').split(",")
                signature_table += '{'
                for arg in name_args:
                    arg_index = args_map[arg]
                    signature_table += f"""{arg_index}, """
                signature_table = signature_table[:-2]
                signature_table += '}, '
                type_num += 1
            signature_table = signature_table[:-2]
    return type_num, signature_table


def gen_pyboost_code(work_path, ops_yaml_data, doc_yaml_data):
    """ gen_pyboost_code """
    # generate pyboost inner prim
    gen_pyboost_inner_prim(work_path, ops_yaml_data)
    # generate pyboost py func
    gen_pyboost_py_func(work_path, ops_yaml_data, doc_yaml_data)
    # generate ops header file
    generate_ops_header_files(work_path, ops_yaml_data)
    # generate pyboost functions
    generate_pyboost_functions(work_path, ops_yaml_data)
    # generate pyboost grad functions
    generate_pyboost_grad_functions(work_path, ops_yaml_data)
    # generate pyboost backend cpp code
    generate_pyboost_op_cpp_code(work_path, ops_yaml_data)
