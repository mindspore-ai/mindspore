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
import pathlib
from .pyboost_utils import get_convert_type_str, get_input_dtype, get_return_type, tuple_input_to_cpp_type, \
    number_input_to_cpp_type, get_const_number_convert, get_tuple_input_convert, get_pyboost_name
from . import pyboost_utils
from .template import CppTemplate
from . import template
from .op_proto import OpProto
from .pyboost_utils import get_disable_flag, get_op_name, py_licence_str


def clean_auto_generate_dir(work_path, relative_path):
    """ clean_auto_generate_dir """
    dir_path = os.path.join(work_path, relative_path)
    if not os.path.exists(dir_path):
        return
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def generate_pyboost_base_op_header_code(work_path, op_name_str, operator_name, call_args_with_type, cpp_func_return):
    """ generate_pyboost_base_op_header_code """
    pyboost_op_header_str = template.PYBOOST_BASE_OP_DEFINE_TEMPLATE.replace(op_name=op_name_str,
                                                                             op_name_upper=op_name_str.upper(),
                                                                             call_args=call_args_with_type,
                                                                             return_type=cpp_func_return)
    op_header_dir_path = os.path.join(work_path, "mindspore/ccsrc/kernel/pyboost/auto_generate/")
    pathlib.Path(op_header_dir_path).mkdir(parents=True, exist_ok=True)
    op_file_path = os.path.join(op_header_dir_path, operator_name + ".h")
    with open(op_file_path, "w") as f:
        f.write(pyboost_op_header_str)


def generate_pyboost_ascend_op_header_code(work_path, op_name_str, operator_name, call_args_with_type, cpp_func_return):
    """ generate_pyboost_ascend_op_header_code """
    pyboost_ascend_op_str = template.PYBOOST_ASCEND_OP_HEADER_TEMPLATE.replace(op_name=op_name_str,
                                                                               op_name_upper=op_name_str.upper(),
                                                                               operator_name=operator_name,
                                                                               call_args_with_type=call_args_with_type,
                                                                               return_type=cpp_func_return)
    ascend_op_header_dir_path = os.path.join(work_path,
                                             "mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/auto_generate/")
    pathlib.Path(ascend_op_header_dir_path).mkdir(parents=True, exist_ok=True)
    ascend_op_file_path = os.path.join(ascend_op_header_dir_path, operator_name + ".h")
    with open(ascend_op_file_path, "w") as f:
        f.write(pyboost_ascend_op_str)


def generate_pyboost_ascend_op_source_code(work_path, pyboost_yaml_data, prim_name_str, operator_name,
                                           proto_operator_name, call_args_type,
                                           call_args_str, op_outputs, call_outputs, call_args_with_type,
                                           cpp_func_return, call_args_after_convert, const_number_convert,
                                           value_tuple_convert, need_malloc_tensors, inplace_process):
    """ generate_pyboost_ascend_op_source_code """
    # PyBoost ascend op source generate
    call_args_tensor = []
    for type, arg_name in zip(call_args_type, call_args_str):
        if type == "TensorPtr" or type == "std::optional<TensorPtr>":
            call_args_tensor.append(arg_name)

    malloc_inputs = ''
    for item in need_malloc_tensors:
        malloc_inputs += f'runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context_, {item}, "{item}");\n'

    # launch mode: cube or not
    # call_impl
    call_impl = ''
    customize_include = ''
    op_desc = pyboost_yaml_data[prim_name_str]['Ascend']
    op_name_str = prim_name_str
    cube_math_type = ''
    get_cube_math_type = ''
    real_output = ', ' + op_outputs
    if prim_name_str.endswith('Ext'):
        op_name_str = prim_name_str[:-3]
    if operator_name.endswith('ext'):
        operator_name = operator_name[:-4]
    if op_desc['mode'] == 'normal':
        if op_desc['cube'] is True:
            get_cube_math_type = "auto cube_math_type = GetCubeMathType();"
            cube_math_type = ', cube_math_type'
        if 'aclnn' in op_desc:
            aclnn_name = op_desc['aclnn']
        else:
            aclnn_name = 'aclnn' + op_name_str
        if inplace_process != '':
            real_output = ''

        call_impl = template.PYBOOST_ASCEND_CALL_TEMPLATE.replace(aclnn_name=aclnn_name,
                                                                  call_args=call_args_str,
                                                                  call_tensors=call_args_tensor,
                                                                  value_tuple_convert=value_tuple_convert,
                                                                  const_number_convert=const_number_convert,
                                                                  malloc_inputs=malloc_inputs,
                                                                  get_cube_math_type=get_cube_math_type,
                                                                  cube_math_type=cube_math_type,
                                                                  aclnn_call_args=call_args_after_convert,
                                                                  return_values=call_outputs,
                                                                  outputs=real_output,
                                                                  inplace_process=inplace_process)
    elif op_desc['mode'] == 'customize':
        call_impl = template.PYBOOST_CUSTOMIZE_CALL_TEMPLATE.replace(op_name=op_name_str,
                                                                     call_args=call_args_str,
                                                                     call_tensors=call_args_tensor,
                                                                     )
        customize_include = "#include \"plugin/device/ascend/kernel/pyboost/customize/{}.h\"".format(
            operator_name.lower())
    elif op_desc['mode'] == 'view':
        call_impl = template.PYBOOST_VIEW_CALL_TEMPLATE.replace(op_name=prim_name_str,
                                                                call_args=call_args_str,
                                                                call_tensors=call_args_tensor,
                                                                input=call_args_str[0])
        customize_include = "#include \"mindspore/core/ops/view/{}_strides_calc.h\"".format(proto_operator_name)
    else:
        raise Exception("Not support mode " + op_desc['mode'])

    pyboost_ascend_op_source_str = \
        template.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE.replace(op_name=op_name_str,
                                                           operator_name=operator_name,
                                                           call_args_with_type=call_args_with_type,
                                                           return_type=cpp_func_return,
                                                           customize_include=customize_include,
                                                           call_impl=call_impl)
    ascend_op_header_dir_path = os.path.join(work_path,
                                             "mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/auto_generate/")
    ascend_op_source_file_path = os.path.join(ascend_op_header_dir_path, operator_name.lower() + ".cc")
    with open(ascend_op_source_file_path, "w") as f:
        f.write(pyboost_ascend_op_source_str)


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
    op_register_file_path = os.path.join(op_register_dir_path, "op_register.cc")
    with open(op_register_file_path, "w") as f:
        f.write(op_register_file_str)


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
    for return_obj in op_proto.returns:
        returns_type.append(get_return_type(return_obj.arg_dtype))
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
            op_outputs = 'outputs_[0]'
            call_outputs = op_outputs
        elif returns_type[0] == "std::vector<tensor::TensorPtr>":
            op_outputs = 'outputs_'
            call_outputs = op_outputs
        else:
            raise Exception("Not support return type {}".format(returns_type[0]))
    elif len(returns_type) > 1:
        outputs_str = ''
        for i in range(len(returns_type)):
            outputs_str += 'outputs_[{}],'.format(i)
        op_outputs = outputs_str[:-1]
        call_outputs = "std::make_tuple(" + op_outputs + ")"

    return op_outputs, call_outputs


def generate_ops_header_files(work_path, yaml_data):
    """
    :param work_path:
    :param yaml:
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
    file_path = os.path.join(dir_path, "gen_ops_def.h")
    with open(file_path, "w") as f:
        f.write(ops_header_file)


def generate_parser_func(op_proto: OpProto) -> str:
    """
    Generate parser func
    :param op_proto:
    :return: str
    """
    convert_template = CppTemplate("auto $arg_name = converter.${convert_func}($arg_index);\n")
    parser_func_str = ''
    for index, arg in enumerate(op_proto.op_args):
        is_optional = (arg.as_init_arg and str(arg.init) == 'None')
        convert_type_str = get_convert_type_str(arg.arg_dtype, is_optional)
        parser_func_str += convert_template.replace(arg_name=arg.arg_name, convert_func=convert_type_str,
                                                    arg_index=pyboost_utils.get_index(index))
    return parser_func_str


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
        if not op_proto.is_pyboost:
            continue
        op_def_name_str = f"g{op_proto.class_name}"
        prim_name_str = op_proto.class_name
        operator_name = op_proto.operator_name
        if operator_name.endswith('ext'):
            operator_name = operator_name[:-4]
        op_name_str = prim_name_str
        if prim_name_str.endswith('Ext'):
            op_name_str = prim_name_str[:-3]
        op_args_str = [op_arg.arg_name for op_arg in op_proto.op_args]
        parser_body_str = generate_parser_func(op_proto)

        convert_to_tensor_template = CppTemplate(
            "auto ${output} = PyNativeAlgo::Common::StubNodeToTensor(${input});\n")
        convert_to_tensor_optional_template = CppTemplate(
            "auto ${output} = PyNativeAlgo::Common::StubNodeToTensorOptional(${input});\n")
        convert_to_tensor_list_template = CppTemplate(
            "auto ${output} = PyNativeAlgo::Common::StubNodeToValueTuple(${input});\n")

        grad_args_str = []
        call_args_str = []
        cast_args_str = []
        convert_stub_str = ''
        optional_to_value_str = ''
        for op_arg in op_proto.op_args:
            grad_arg = ''
            call_arg = ''
            cast_arg = ''
            if pyboost_utils.is_tensor(op_arg):
                if op_arg.as_init_arg and str(op_arg.init) == 'None':
                    convert_stub_output_name = op_arg.arg_name + '_optional'
                    convert_stub_str += convert_to_tensor_optional_template.replace(output=convert_stub_output_name,
                                                                                    input=op_arg.arg_name)
                    cast_output = 'cast_' + convert_stub_output_name
                    convert_optional_to_value_template = CppTemplate(
                        "auto ${output} = PyNativeAlgo::PyBoost::OptionalToValue(${input});\n")
                    convert_optional_to_value_name = op_arg.arg_name + "_value"
                    optional_to_value_str += convert_optional_to_value_template.replace(input=cast_output,
                                                                                        output=convert_optional_to_value_name)
                    call_arg = convert_stub_output_name
                    grad_arg = convert_optional_to_value_name
                    cast_arg = cast_output
                else:
                    convert_stub_output_name = op_arg.arg_name + "_tensor"
                    convert_stub_str += convert_to_tensor_template.replace(input=op_arg.arg_name,
                                                                           output=convert_stub_output_name)
                    call_arg = convert_stub_output_name
                    grad_arg = "cast_" + convert_stub_output_name
                    cast_arg = grad_arg
            elif pyboost_utils.is_tensor_list(op_arg):
                convert_stub_output_name = op_arg.arg_name + "_tensor_list"
                convert_stub_str += convert_to_tensor_list_template.replace(input=op_arg.arg_name,
                                                                            output=convert_stub_output_name)
                call_arg = convert_stub_output_name
                grad_arg = "cast_" + convert_stub_output_name
                cast_arg = grad_arg
            else:
                call_arg = op_arg.arg_name
                grad_arg = "cast_" + op_arg.arg_name
                cast_arg = grad_arg
            grad_args_str.append(grad_arg)
            call_args_str.append(call_arg)
            cast_args_str.append(cast_arg)
        pyboost_func_str += template.PYBOOST_FUNCTION_TEMPLATE.replace(func_name=op_proto.pyboost_function_name,
                                                                       op_def_name=op_def_name_str,
                                                                       parser_body=parser_body_str, op_name=op_name_str,
                                                                       convert_stub=convert_stub_str,
                                                                       optional_to_value=optional_to_value_str,
                                                                       call_args=call_args_str,
                                                                       grad_args=grad_args_str,
                                                                       cast_args=cast_args_str,
                                                                       op_args=op_args_str)
        pyboost_func_str = pyboost_func_str + template.NEW_LINE + template.NEW_LINE
        pyboost_func_pybind_def += template.REGISTER_DEFINE_TEMPLATE.replace(
            pyboost_op_name=get_pyboost_name(op_proto.operator_name),
            pyboost_cfunc_name=op_proto.pyboost_function_name)
        pyboost_func_include_headers_str += pyboost_func_include_header_template.replace(operator_name=operator_name)
    register_func_str = template.REGISTER_TEMPLATE.replace(register_func=pyboost_func_pybind_def)

    pyboost_func_file = template.PYBOOST_HEADER_TEMPLATE.replace(include_op_header=pyboost_func_include_headers_str,
                                                                 function_body=pyboost_func_str,
                                                                 register_function_body=register_func_str)
    dir_path = os.path.join(work_path, "mindspore/ccsrc/pipeline/pynative/op_function/auto_generate")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(dir_path, "pyboost_functions.cc")
    with open(file_path, "w") as f:
        f.write(pyboost_func_file)


def generate_inplace_process_cpp_code(op_proto):
    """ generate_ref_process_cpp_code """
    inplace_process = f'// RefOps update output by input tensor\n'
    has_ref = False
    for index, return_obj in enumerate(op_proto.returns):
        if return_obj.inplace == '':
            continue
        has_ref = True
        inplace_process += f'outputs_[{index}]->set_device_address({return_obj.inplace}_tensor->device_address());\n'
    if has_ref:
        return inplace_process
    else:
        return ''


def generate_pyboost_op_cpp_code(work_path, yaml_data, pyboost_yaml_data):
    """
    Generate pyboost op cpp code from yaml.
    """
    clean_auto_generate_dir(work_path, "mindspore/ccsrc/kernel/pyboost/auto_generate/")
    clean_auto_generate_dir(work_path, "mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/auto_generate/")

    all_op_names = []
    all_operator_names = []
    for operator_name, operator_data in yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        prim_name_str = op_proto.class_name
        if prim_name_str not in pyboost_yaml_data.keys():
            continue
        operator_name = op_proto.operator_name
        if operator_name.endswith('ext'):
            operator_name = operator_name[:-4]
        op_name_str = prim_name_str
        if prim_name_str.endswith('Ext'):
            op_name_str = prim_name_str[:-3]
        call_args_str = []
        call_args_after_convert = []
        call_args_type = []
        value_tuple_convert = []
        const_number_convert = []
        need_malloc_tensors = []
        for op_arg in op_proto.op_args:
            call_arg = ''
            if pyboost_utils.is_tensor(op_arg):
                call_arg = op_arg.arg_name + "_tensor"
                need_malloc_tensors.append(call_arg)
            elif pyboost_utils.is_tensor_list(op_arg):
                call_arg = op_arg.arg_name + "_tensor_list"
            else:
                call_arg = op_arg.arg_name
            call_args_str.append(call_arg)
            is_optional = False
            if op_arg.as_init_arg and str(op_arg.init) == 'None':
                is_optional = True
            call_args_type.append(get_input_dtype(op_arg.arg_dtype, is_optional))
            if number_input_to_cpp_type(op_arg.arg_dtype):
                call_args_after_convert.append(call_arg + "_imm")
                const_number_convert.append(get_const_number_convert(call_arg, op_arg.arg_dtype))
            elif tuple_input_to_cpp_type(op_arg.arg_dtype):
                call_args_after_convert.append(call_arg + "_vector")
                value_tuple_convert.append(get_tuple_input_convert(call_arg, op_arg.arg_dtype))
                if pyboost_utils.is_tensor_list(op_arg):
                    need_malloc_tensors.append(call_arg + "_vector")
            else:
                call_args_after_convert.append(call_arg)

        all_op_names.append(op_name_str)
        all_operator_names.append(operator_name)

        call_args_with_type = []
        for type, arg_name in zip(call_args_type, call_args_str):
            call_args_with_type.append("const " + type + " &" + arg_name)

        cpp_func_return = generate_pyboost_op_func_return_type(op_proto)
        op_outputs, call_func_outputs = generate_pyboost_outputs(op_proto)
        inplace_process = generate_inplace_process_cpp_code(op_proto)

        generate_pyboost_base_op_header_code(work_path, op_name_str, operator_name, call_args_with_type,
                                             cpp_func_return)
        generate_pyboost_ascend_op_header_code(work_path, op_name_str, operator_name, call_args_with_type,
                                               cpp_func_return)
        generate_pyboost_ascend_op_source_code(work_path, pyboost_yaml_data, prim_name_str,
                                               operator_name, op_proto.operator_name, call_args_type, call_args_str,
                                               op_outputs, call_func_outputs, call_args_with_type, cpp_func_return,
                                               call_args_after_convert, const_number_convert, value_tuple_convert,
                                               need_malloc_tensors, inplace_process)
    generate_pyboost_op_register_source_code(work_path, all_op_names, all_operator_names)


def gen_pyboost_py_func(work_path, op_yaml_data, doc_data, pyboost_yaml_data):
    """ gen_pyboost_py_func """
    gen_py = ''
    gen_py += py_licence_str
    op_desc_dict = {}
    for operator_name, operator_desc in doc_data.items():
        desc = operator_desc.get("description")
        op_desc_dict[operator_name] = desc
    gen_py += template.PYBOOST_PY_FUNC_HEADEAR
    func_def = None
    for operator_name, operator_data in op_yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        if op_proto.class_name in pyboost_yaml_data.keys():
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

        description = op_desc_dict.get(operator_name)
        args = operator_data.get('args')
        class_name = get_op_name(operator_name, operator_data.get('class'))
        func_args = []
        init_args = []
        input_args = []
        for arg_name, arg_info in args.items():
            init_value = arg_info.get('init')

            if init_value is None:
                default_value = arg_info.get('default')
                default_value = '=' + default_value if default_value else ''
                func_args.append(arg_name + default_value)
                input_args.append(arg_name)
            else:
                if init_value == 'NO_VALUE':
                    func_args.append(f"""{arg_name}""")
                    init_args.append(arg_name)
                else:
                    func_args.append(f"""{arg_name}={init_value}""")
                    init_args.append(arg_name)
        gen_py += template.PYBOOST_PY_FUNC_TEMPLATE.replace(func_name=func_name, description=description,
                                                            func_args=func_args,
                                                            init_args=init_args,
                                                            operator_name=operator_name,
                                                            class_name=class_name, input_args=input_args)
    dir_path = os.path.join(work_path, "mindspore/python/mindspore/ops/auto_generate")
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(dir_path, "gen_pyboost_func.py")
    with open(file_path, "w") as f:
        f.write(gen_py)


def gen_pyboost_code(work_path, ops_yaml_data, doc_yaml_data, pyboost_yaml_data):
    """ gen_pyboost_code """
    # generate pyboost py func
    gen_pyboost_py_func(work_path, ops_yaml_data, doc_yaml_data, pyboost_yaml_data)
    # generate ops header file
    generate_ops_header_files(work_path, ops_yaml_data)
    # generate pyboost functions
    generate_pyboost_functions(work_path, ops_yaml_data)
    # generate pyboost backend cpp code
    generate_pyboost_op_cpp_code(work_path, ops_yaml_data, pyboost_yaml_data)
