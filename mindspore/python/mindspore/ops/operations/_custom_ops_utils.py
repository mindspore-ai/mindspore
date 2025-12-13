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

"""compile custom kernel with ninja"""

import os
import re
import shlex
import subprocess
import sysconfig
import time
import stat
import json
from mindspore import log as logger
from mindspore.ops import CustomRegOp
from mindspore._c_expression import MSContext

OP_INFO_KEY_INPUT = "input"
OP_INFO_KEY_OUTPUT = "output"
OP_INFO_KEY_LIST = "list"
OP_INFO_KEY_ATTR = "attr"
OP_INFO_KEY_DTYPE = "dtype"
OP_INFO_KEY_FORMAT = "format"

REG_INFO_KEY_INPUTS = "inputs"
REG_INFO_KEY_OUTPUTS = "outputs"
REG_INFO_KEY_ATTRS = "attrs"
REG_INFO_KEY_ATTR = "attr"
REG_INFO_KEY_NAME = "name"
REG_INFO_KEY_PARAM_TYPE = "paramType"
REG_INFO_KEY_VALUE = "value"
REG_INFO_KEY_TYPE = "type"


class VersionManager:
    """version manager"""

    def __init__(self):
        self.entries = {}  # module_name : (version, hash)

    def _get_version(self, module_name):
        """get version"""
        return self.entries.get(module_name, (None, None))[0]

    def _update_version_if_changed(self, module_name, sources, build_args, build_dir):
        """update version if changed"""
        hash_value = self._update_hash(0, build_dir)
        hash_value = self._update_sources_hash(hash_value, sources)
        hash_value = self._update_args_hash(hash_value, build_args)

        entry = self.entries.get(module_name)
        if entry is None:
            self.entries[module_name] = entry = (0, hash_value)
        elif hash_value != entry[1]:
            self.entries[module_name] = entry = (entry[0] + 1, hash_value)

        return entry[0]

    @staticmethod
    def _update_hash(seed, value):
        """update hash value"""
        # Good old boost::hash_combine
        return seed ^ (hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2))

    def _update_sources_hash(self, hash_value, sources):
        """hash source files"""
        for filename in sources:
            with open(filename) as file:
                hash_value = self._update_hash(hash_value, file.read())
        return hash_value

    def _update_args_hash(self, hash_value, build_args):
        """hash build arguments"""
        for group in build_args:
            if group:
                for argument in group:
                    hash_value = self._update_hash(hash_value, argument)
        return hash_value

    def check_version(self, name, sources, cflags, ldflags, include_paths, build_dir):
        """check version"""
        old_version = self._get_version(name)
        version = self._update_version_if_changed(name, sources, [cflags, ldflags, include_paths], build_dir)
        logger.info(f'Build module {name}, version={version}')
        if version > 0:
            if version != old_version:
                logger.info(
                    f'The conditions for extension module {name} have changed. '
                    f'Updating to version {version} and re-building as {name}_v{version}.'
                )
            name = f'{name}_v{version}'

        if version != old_version:
            return True
        logger.info(f'No modifications detected for extension module {name}')
        return False


version_manager = VersionManager()


class FileLocker:
    """FileLocker"""

    def __init__(self, build_dir):
        """FileLocker"""
        self.lock_file_name = os.path.join(build_dir, 'build.lock')
        self.lock_fd = None

    def try_lock(self):
        """Acquire a file-based lock."""
        try:
            mode = stat.S_IRUSR | stat.S_IWUSR
            self.lock_fd = os.open(self.lock_file_name, os.O_CREAT | os.O_EXCL, mode)
            return True
        except FileExistsError:
            return False

    def release_lock(self):
        """Release the file-based lock."""
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        os.remove(self.lock_file_name)

    def wait(self):
        """Wait until lock is released."""
        while os.path.exists(self.lock_file_name):
            time.sleep(0.5)


class ExtensionBuilder:
    """ExtensionBuilder"""

    def __init__(self, build_dir):
        """ExtensionBuilder"""
        self.build_dir = build_dir

    def _compile(self, name, sources, cflags, ldflags, include_paths):
        """Compile."""
        if version_manager.check_version(name, sources, cflags, ldflags, include_paths, self.build_dir):
            locker = FileLocker(self.build_dir)
            if locker.try_lock():
                try:
                    self._write_ninja_file_and_build_library(name, sources, cflags, ldflags, include_paths)
                finally:
                    locker.release_lock()
            else:
                locker.wait()
        logger.info(f'Loading extension module {name}...')

    @staticmethod
    def _verify_ninja_availability():
        """Check ninja is available."""
        try:
            subprocess.check_output('ninja --version'.split())
        except Exception as e:
            msg = (
                "Ninja is required to load C++ extensions. "
                "You can install it with: pip install ninja"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _write_ninja_file_and_build_library(self, module_name, sources, cflags, ldflags, include_paths):
        """Write ninja file and build library."""
        self._verify_ninja_availability()

        ninja_build_file = os.path.join(self.build_dir, 'build.ninja')
        logger.info(f'Save ninja build file {ninja_build_file}.')
        self._write_ninja_file(ninja_build_file, module_name, sources, cflags, ldflags, include_paths)

        logger.info(f'Building extension module {module_name}.')
        self._run_ninja_build(module_name)

    @staticmethod
    def _write_ninja_file(fname, name, sources, extra_cflags, extra_ldflags, extra_include_paths):
        """Write ninja file."""
        python_include_path = sysconfig.get_path('include', scheme='posix_prefix')
        python_includes = [python_include_path] if python_include_path is not None else []
        cflags = []
        cflags += [f'-I{shlex.quote(os.path.abspath(include.strip()))}' for include in extra_include_paths]
        cflags += [f'-isystem {shlex.quote(include)}' for include in python_includes]
        cflags += extra_cflags
        cflags = [flag.strip() for flag in cflags]

        # '/path/to/file.cpp' -> 'file'
        objs = [os.path.splitext(os.path.basename(src))[0] + ".o" for src in sources]
        sources = [os.path.abspath(file) for file in sources]
        ldflags = [flag.strip() for flag in extra_ldflags]
        target = name + '.so'

        config = ['ninja_required_version = 1.3']
        config.append('cxx = ' + os.environ.get('CXX', 'g++'))

        flags = [f'cflags = {" ".join(cflags)}']
        flags.append(f'ldflags = {" ".join(ldflags)}')

        compile_rule = ['rule compile']
        compile_rule.append('  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

        build = [f'build {obj.replace(" ", "$ ")}: compile {src.replace(" ", "$ ")}' for src, obj in zip(sources, objs)]

        link_rule = ['rule link', '  command = $cxx $in $ldflags -o $out']
        link = [f'build {target}: link {" ".join(objs)}']
        default = [f'default {target}']

        blocks = [config, flags, compile_rule, link_rule, build, link, default]
        content = "\n\n".join("\n".join(b) for b in blocks) + "\n"

        if os.path.exists(fname):
            with open(fname) as f:
                old_content = f.read()
            if old_content == content:
                return

        with open(fname, 'w') as source_file:
            source_file.write(content)

    def _run_ninja_build(self, module_name):
        """Run ninja build and log output to .build_log.txt"""
        cmd = ['ninja', '-v']
        env = os.environ.copy()
        log_file = os.path.join(self.build_dir, '.build_log.txt')

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                # If the build succeeds, do nothing with the output (silent)
                subprocess.run(cmd, stdout=f, stderr=f, cwd=self.build_dir, check=True, env=env)
        except subprocess.CalledProcessError as e:
            with open(log_file, 'r', encoding='utf-8') as rf:
                full_output = rf.read()
            msg = f"Error building extension '{module_name}': {full_output}"

            # In multi-card situation, only one process build the library.
            # When building failed, the old extension library should be removed.
            so_file = os.path.join(self.build_dir, f"{module_name}.so")
            if os.path.exists(so_file):
                os.remove(so_file)
            logger.error(msg)
            raise RuntimeError(msg) from e

    def build(self, module_name, sources, extra_cflags=None, extra_ldflags=None, extra_include_paths=None):
        """Build module."""
        src = [sources] if isinstance(sources, str) else sources
        self._compile(module_name, src, extra_cflags, extra_ldflags, extra_include_paths)
        return os.path.join(self.build_dir, f"{module_name}.so")


class CustomCodeGenerator:
    """A class to generate custom C++ code based on input and output types."""

    def __init__(self):
        """Initialize the CustomCodeGenerator with the header content."""
        self.header = """
#include <vector>
#include "acl/acl_base.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;
                                """.strip()

        self.supported_input_scalar_type = ['int64_t', 'uint64_t', 'float', 'double', 'bool', 'aclDataType']
        self.supported_input_pointer_type = ['aclTensor*', 'aclScalar*', 'aclIntArray*', 'aclFloatArray*',
                                             'aclBoolArray*',
                                             'aclTensorList*']
        self.supported_input_type = self.supported_input_pointer_type + self.supported_input_scalar_type
        self.supported_output_type = ["aclTensor*", "aclTensorList*"]

    def _get_input_output_types(self, reg_info):
        """
        Extracts input and output types from registration information.

        Args:
            reg_info (dict): Registration information containing input, output, and attribute details.

        Returns:
            tuple: A tuple containing two lists, the first being input types and the second being output types.
        """
        inputs = reg_info.get(REG_INFO_KEY_INPUTS, [])
        outputs = reg_info.get(REG_INFO_KEY_OUTPUTS, [])
        attrs = reg_info.get(REG_INFO_KEY_ATTR, [])

        inputs_types = []
        outputs_types = []
        for input in inputs:
            if input.get(REG_INFO_KEY_PARAM_TYPE) == "dynamic":
                inputs_types.append("aclTensorList*")
            else:
                inputs_types.append("aclTensor*")
        for attr in attrs:
            inputs_types.append(CustomCodeGenerator._get_type_declaration(attr.get(REG_INFO_KEY_TYPE)))

        for output in outputs:
            if output.get(REG_INFO_KEY_PARAM_TYPE) == "dynamic":
                outputs_types.append("aclTensorList*")
            else:
                outputs_types.append("aclTensor*")

        return inputs_types, outputs_types

    def get_api_types_by_reg_info(self, reg_info):
        """
        Retrieves API types based on registration information.

        Combines input types, output types, and additional parameter types.

        Args:
            reg_info (dict): Registration information.

        Returns:
            list: A list of API types.
        """
        inputs_types, outputs_types = self._get_input_output_types(reg_info)
        return inputs_types + outputs_types + ['int64_t*', 'aclOpExecutor**']

    def generate_callback_by_reg_info(self, func_name, reg_info):
        """
        Generates a callback function based on registration information.

        Args:
            func_name (str): Name of the function.
            reg_info (dict): Registration information.

        Returns:
            str: Generated callback code.
        """
        inputs_types, outputs_types = self._get_input_output_types(reg_info)
        return self._generate_callback(func_name, inputs_types, outputs_types)

    def generate_callback_by_types(self, func_name, reg_info, input_output_types):
        """
        Generates a callback function based on types and registration information.

        Validates the consistency between registration info and input/output types.

        Args:
            func_name (str): Name of the function.
            reg_info (dict): Registration information.
            input_output_types (list): List of input and output types.

        Returns:
            str: Generated callback code.

        Raises:
            RuntimeError: If there's inconsistency between reg info and input/output types.
        """
        inputs_types, outputs_types = self._get_input_output_types(reg_info)
        input_size = len(inputs_types)
        output_size = len(outputs_types)
        aclnn_api_input_size = len(input_output_types)
        func_params_len = input_size + output_size + 2
        if func_params_len != len(input_output_types):
            raise RuntimeError(
                f"Reg info input size: {func_params_len} is not equal to aclnn api input size {aclnn_api_input_size}")
        reg_info_input_output_type = inputs_types + outputs_types
        for i, typ in enumerate(reg_info_input_output_type):
            if typ != input_output_types[i]:
                logger.warning(
                    "Reg info type {} is not  same with function prototype {}".format(typ,
                                                                                      input_output_types[i]))

        return self._generate_callback(func_name, input_output_types[:input_size],
                                       input_output_types[input_size:output_size + input_size])

    def _generate_callback_inputs(self, inputs_types):
        """
        Generates code for callback inputs based on input types.

        Args:
            inputs_types (list): List of input types.

        Returns:
            list: List of generated input code strings.

        Raises:
            RuntimeError: If unsupported input type is encountered.
        """
        inputs_code = []

        for i, typ in enumerate(inputs_types):
            if typ not in self.supported_input_type:
                raise RuntimeError(
                    f"Unsupported input type: {typ}, supported input types are: {self.supported_input_type}")

            if typ in self.supported_input_scalar_type:
                typ = typ + "*"
            inputs_code.append(f"  {typ} input{i} = static_cast<{typ}>(inputs[{i}])")
        return inputs_code

    def _generate_callback_outputs(self, outputs_types):
        """
        Generates code for callback outputs based on output types.

        Args:
            outputs_types (list): List of output types.

        Returns:
            list: List of generated output code strings.

        Raises:
            RuntimeError: If unsupported output type is encountered.
        """
        outputs_code = []
        for i, typ in enumerate(outputs_types):
            if typ not in self.supported_output_type:
                raise RuntimeError(
                    f"Unsupported output type: {typ}, supported output types are: {self.supported_output_type}")
            outputs_code.append(f"  {typ} output{i} = static_cast<{typ}>(outputs[{i}])")
        return outputs_code

    def _generate_callback_func_params(self, inputs_types, outputs_types):
        """
        Generates function parameters for callback based on input and output types.

        Args:
            inputs_types (list): List of input types.
            outputs_types (list): List of output types.

        Returns:
            list: List of generated function parameter strings.

        Raises:
            RuntimeError: If unsupported input or output type is encountered.
        """
        func_params = []
        for i, _ in enumerate(inputs_types):
            typ = inputs_types[i]
            if typ in self.supported_input_pointer_type:
                func_params.append(f"input{i}")
            elif typ in self.supported_input_scalar_type:
                func_params.append(f"*input{i}")
            else:
                raise RuntimeError(
                    f"Unsupported input type: {typ}, supported input types are: {self.supported_input_type}")

        for i, _ in enumerate(outputs_types):
            typ = outputs_types[i]
            if typ in self.supported_output_type:
                func_params.append(f"output{i}")
            else:
                raise RuntimeError(
                    f"Unsupported output type: {typ}, supported output types are: {self.supported_output_type}")

        func_params.append("workspace_size")
        func_params.append("executor")
        return func_params

    def _generate_callback(self, func_name, inputs_types, outputs_types):
        """Generate C++ code based on the provided function name, input types, and output types.

        Args:
            func_name (str): The name of the function to generate.
            inputs_types (str): A comma-separated string of input types.
            outputs_types (str): A comma-separated string of output types.

        Returns:
            str: The generated C++ callback func.
        """

        inputs_code = self._generate_callback_inputs(inputs_types)
        outputs_code = self._generate_callback_outputs(outputs_types)
        func_params = self._generate_callback_func_params(inputs_types, outputs_types)

        input_declarations = ', '.join(inputs_types)
        output_declarations = ', '.join(outputs_types)

        code = """
{header}

extern "C" int {func_name}GetWorkSpaceSize(void *func_ptr, std::vector<void *> inputs, std::vector<void *> outputs,
                           uint64_t *workspace_size, aclOpExecutor **executor) {{
  using FuncType = int (*)({input_declarations}, {output_declarations}, uint64_t *, aclOpExecutor **);
  auto func = reinterpret_cast<FuncType>(func_ptr);
{inputs_code}
{outputs_code}
  return func({func_params});
}}""".format(
    header=self.header, func_name=func_name, input_declarations=input_declarations,
    output_declarations=output_declarations, inputs_code=";\n".join(inputs_code) + ";",
    outputs_code=";\n".join(outputs_code) + ";", func_params=", ".join(func_params))
        return code

    @staticmethod
    def _get_type_declaration(typ):
        """Get the C++ type declaration based on the type.

       Args:
           typ (str): The type for which to get the declaration.

       Returns:
           str: The C++ type declaration.
       """
        type_map = {
            "tensor": "aclTensor*",
            "int": "int64_t",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "number": "aclScalar*",
            "listInt": "aclIntArray*",
            "listBool": "aclBoolArray*",
            "listFloat": "aclFloatArray*"
        }
        try:
            return type_map[typ]
        except KeyError as e:
            raise RuntimeError(f"Unsupported type: {typ}") from e


class CustomInfoGenerator:
    """
    A utility class for generating custom operator registration information.

    This class is designed to parse operator configuration from JSON files and
    generate registration information compatible with the Ascend platform.
    """

    def __init__(self, op_name):
        """
        Initialize a new instance of CustomInfoGenerator.

        Args:
            op_name (str): Name of the operator to generate registration info for.
        """
        self.ori_op_name = op_name
        self.prefix = "aclnn"
        self.pure_op_name = self._get_pure_name(op_name)
        self.prefix_op_name = self._get_prefix_name(op_name)
        self.aclnn_api_file_name = CustomInfoGenerator._get_aclnn_api_file_name(self.prefix_op_name)

        self.env_ascend_opp_path = os.getenv("ASCEND_OPP_PATH")
        self.env_ascend_custom_opp_path = os.getenv("ASCEND_CUSTOM_OPP_PATH")

        self.op_info_paths = []
        self.target_json_path = ""
        self.op_info = ""

        self.aclnn_api_paths = []
        self.aclnn_api = ""

    @staticmethod
    def _get_aclnn_api_file_name(op_name):
        """ Converts a camel-case operation name to an underscore-separated filename with a .h suffix."""
        name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', op_name)
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower() + ".h"

    def _get_pure_name(self, op_name):
        """Remove the "aclnn_" prefix from the operator name if it exists."""
        if not op_name:
            raise ValueError("op_name cannot be None or empty")

        if op_name.startswith(self.prefix):
            return op_name[len(self.prefix):]
        return op_name

    def _get_prefix_name(self, op_name):
        """Add the "aclnn" prefix to the operator name if it doesn't already exist."""
        if not op_name:
            raise ValueError("op_name cannot be None or empty")

        if op_name.startswith(self.prefix):
            return op_name
        return self.prefix + op_name

    def _get_aclnn_api_from_file(self, dir_path):
        """
        Attempts to extract the AclNN API content from a specified directory path.

        Args:
            dir_path (str): Directory path to search for the AclNN API file.

        Returns:
            bool: True if the API content is successfully extracted, False otherwise.

        Raises:
            ValueError: If the start or end marker for the API function is not found.
        """
        self.aclnn_api_paths.append(dir_path)
        file_path = os.path.join(dir_path, self.aclnn_api_file_name)
        if not os.path.exists(file_path):
            return False

        with open(file_path, 'r') as file:
            content = file.read()

            func_marker = self.prefix_op_name + "GetWorkspaceSize("
            end_marker = ");"

            start_pos = 0
            func_pos = -1
            while True:
                func_pos = content.find(func_marker, start_pos)
                if func_pos == -1:
                    break

                if re.search(r'(\s+|\n|^)\baclnnStatus\b\s*$', content[:func_pos]):
                    break

                start_pos = func_pos + 1

            if func_pos == -1:
                raise ValueError(
                    f"Can not find 'aclnnStatus' followed by function "
                    f"[{func_marker}] in file [{file_path}]"
                )

            end_pos = content.find(end_marker, func_pos)
            if end_pos == -1:
                raise ValueError(f"Can not find function [{start_marker}] in file [{file_path}]")

            self.aclnn_api = content[func_pos:end_pos + len(end_marker)]
            return True

    def _get_aclnn_api_params(self):
        """
        Searches for the AaclNN API file in multiple predefined paths and extracts its content.

        Raises:
            RuntimeError: If the AclNN API file is not found in any of the specified paths.
        """
        if self.env_ascend_custom_opp_path is not None:
            custom_opp_paths = self.env_ascend_custom_opp_path.split(":")
            for custom_opp_path in custom_opp_paths:
                aclnn_api_file_path = os.path.join(custom_opp_path, "op_api/include/")
                if self._get_aclnn_api_from_file(aclnn_api_file_path):
                    return

        opp_vendors_path = os.path.join(self.env_ascend_opp_path, "vendors")
        opp_vendors_config_path = os.path.join(opp_vendors_path, "config.ini")
        if os.path.exists(opp_vendors_config_path):
            priorities = CustomInfoGenerator._parse_load_priority(opp_vendors_config_path)
            for priority in priorities:
                aclnn_api_file_path = os.path.join(opp_vendors_path, priority.strip(), "op_api/include/")
                if self._get_aclnn_api_from_file(aclnn_api_file_path):
                    return
        aclnn_api_file_path = os.path.join(self.env_ascend_opp_path, "../include/aclnnop")
        if self._get_aclnn_api_from_file(aclnn_api_file_path):
            return

        paths = ",".join(str(item) for item in self.aclnn_api_paths)
        logger.warning(f"Cannot find file [{self.aclnn_api_file_name}] in paths [{paths}]")

    def get_aclnn_api_types(self):
        """
        Extracts and returns the input types from the AclNN API function declaration.

        Args:
            None

        Returns:
            list: A list of input types extracted from the AclNN API function.

        Raises:
            RuntimeError: If the AclNN API content is empty or if the input types cannot be parsed.
        """
        self._get_aclnn_api_params()
        param_types = []
        if self.aclnn_api == "":
            return param_types

        # step1: get string by '()'
        param_section = re.search(r'\((.*?)\)', self.aclnn_api, re.DOTALL).group(1)

        # step2: split by ','
        params = re.split(r',\s*', param_section)

        # step3: get type
        for param in params:
            param = param.replace('const ', '')
            type_part = re.search(r'^\s*(\w+\s*\*+|\w+)', param).group(1)
            type_part = type_part.replace(' ', '')
            param_types.append(type_part)
        return param_types

    @staticmethod
    def _parse_load_priority(config_file_path):
        """
        Parse the load priority from a configuration file.

        Extracts the load_priority configuration item from the specified file
        and splits it into a list of priorities.

        Args:
            config_file_path (str): Path to the configuration file.

        Returns:
            list: List of load priorities.
        """
        load_priority = ''
        with open(config_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('load_priority'):
                    # Parse key-value pair from the line
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key.lower() == 'load_priority':
                        load_priority = value
                        break
        # Split the priority string into a list
        priorities = [item.strip() for item in load_priority.split(',') if item.strip()]
        return priorities

    def _get_op_info_from_file(self, op_info_path):
        """
        Retrieve operator information from a JSON file.

        Checks if the specified JSON file exists and attempts to extract
        operator information from it. If the operator is found, records
        the file path and operator information.

        Args:
            op_info_path (str): Path to the JSON file.

        Returns:
            bool: Whether the operator information was found.
        """
        self.op_info_paths.append(op_info_path)
        if not os.path.exists(op_info_path):
            logger.debug("Custom config path not found: %s", op_info_path)
            return False

        with open(op_info_path, 'r', encoding='utf-8') as file:
            op_json_data = json.load(file)

        if self.pure_op_name in op_json_data:
            self.target_json_path = op_info_path
            self.op_info = op_json_data[self.pure_op_name]
            return True
        return False

    def _get_op_json(self):
        """
        Retrieve the JSON configuration for the operator.

        Searches for the operator's JSON configuration in the following order:
        1. Custom OPP path (from ASCEND_CUSTOM_OPP_PATH environment variable)
        2. Vendor OPP path (from ASCEND_OPP_PATH's vendors directory)
        3. Built-in OPP path (from ASCEND_OPP_PATH's built-in directory)

        Raises:
            RuntimeError: If the operator is not found in any JSON file.
        """

        soc_version = MSContext.get_instance().get_ascend_soc_version()
        op_info_json_prefix = f"aic-{soc_version}-ops-info"
        op_info_json_subdir = os.path.join("op_impl/ai_core/tbe/config", soc_version)

        if self.env_ascend_custom_opp_path is not None:
            custom_opp_paths = self.env_ascend_custom_opp_path.split(":")
            for custom_opp_path in custom_opp_paths:
                op_info_path = os.path.join(custom_opp_path, op_info_json_subdir, f"{op_info_json_prefix}.json")
                if self._get_op_info_from_file(op_info_path):
                    return

        opp_vendors_path = os.path.join(self.env_ascend_opp_path, "vendors")
        opp_vendors_config_path = os.path.join(opp_vendors_path, "config.ini")
        if os.path.exists(opp_vendors_config_path):
            priorities = CustomInfoGenerator._parse_load_priority(opp_vendors_config_path)
            for priority in priorities:
                op_info_path = os.path.join(opp_vendors_path, priority.strip(), op_info_json_subdir,
                                            f"{op_info_json_prefix}.json")
                if self._get_op_info_from_file(op_info_path):
                    return

        builtin_dir = os.path.join(self.env_ascend_opp_path, "built-in", op_info_json_subdir)
        for fname in os.listdir(builtin_dir):
            if fname.startswith(op_info_json_prefix) and fname.endswith(".json"):
                op_info_path = os.path.join(builtin_dir, fname)
                if self._get_op_info_from_file(op_info_path):
                    return

        paths = ",".join(str(item) for item in self.op_info_paths)
        raise RuntimeError(f"Cannot find operator [{self.pure_op_name}] in JSON files [{paths}]")

    def _generate_reg_info(self):
        """
        Generate registration information for the operator.

        Extracts input, output, and attribute information from the parsed
        operator data and constructs a registration information dictionary.

        Returns:
            dict: Registration information for the operator.
        """
        self._get_op_json()

        inputs = []
        outputs = []

        # Extract input and output information
        for key in sorted(self.op_info.keys()):
            if key.startswith(OP_INFO_KEY_INPUT):
                inputs.append(self.op_info[key])
            elif key.startswith(OP_INFO_KEY_OUTPUT):
                outputs.append(self.op_info[key])

        attrs = []
        # Process attributes if available
        if (OP_INFO_KEY_ATTR in self.op_info and
                OP_INFO_KEY_LIST in self.op_info[OP_INFO_KEY_ATTR]):
            attr_list = self.op_info[OP_INFO_KEY_ATTR][OP_INFO_KEY_LIST]
            for attr in attr_list.split(","):
                attr_key = f"attr_{attr}"
                if attr_key in self.op_info:
                    attr_info = self.op_info[attr_key]
                    attr_info[REG_INFO_KEY_NAME] = attr
                    attrs.append(attr_info)
                else:
                    raise KeyError(
                        f"Attr key '{attr_key}' not found in operator '{self.pure_op_name}' "
                        f"from JSON file '{self.target_json_path}'")

        reg_info = {
            REG_INFO_KEY_INPUTS: inputs,
            REG_INFO_KEY_OUTPUTS: outputs,
            REG_INFO_KEY_ATTRS: attrs
        }

        return reg_info

    @staticmethod
    def _get_dtype_format(dtype, format_str):
        """
        Combine data type and format into a tuple.

        Args:
            dtype (str): Data type string.
            format_str (str): Format string.

        Returns:
            tuple: (dtype, format) tuple.
        """
        if dtype == "float":
            dtype = "float32"
        if format_str == "ND":
            format_str = "DefaultFormat"
        return (dtype, format_str)

    def generate_custom_reg_op(self):
        """
        Generate a custom registered operator based on the registration info.

        Constructs a CustomRegOp instance with inputs, outputs, and attributes
        populated from the parsed operator information.

        Returns:
            dict: Registered operator information.
        """
        reg_info = self._generate_reg_info()
        custom_reg_op = CustomRegOp(self.ori_op_name)

        # Process inputs
        inputs_types = []
        inputs_formats = []
        for i, input_data in enumerate(reg_info[REG_INFO_KEY_INPUTS]):
            inputs_types.append(input_data.get(OP_INFO_KEY_DTYPE, "float16").split(","))
            inputs_formats.append(input_data.get(OP_INFO_KEY_FORMAT, "DefaultFormat").split(","))
            custom_reg_op.input(i, input_data[REG_INFO_KEY_NAME], input_data[REG_INFO_KEY_PARAM_TYPE])

        # Process outputs
        outputs_types = []
        outputs_formats = []
        for i, output_data in enumerate(reg_info[REG_INFO_KEY_OUTPUTS]):
            outputs_types.append(output_data.get(OP_INFO_KEY_DTYPE, "float16").split(","))
            outputs_formats.append(output_data.get(OP_INFO_KEY_FORMAT, "DefaultFormat").split(","))
            custom_reg_op.output(i, output_data[REG_INFO_KEY_NAME], output_data[REG_INFO_KEY_PARAM_TYPE])

        # Process attributes
        for attr_data in reg_info[REG_INFO_KEY_ATTRS]:
            custom_reg_op.attr(
                attr_data.get(REG_INFO_KEY_NAME, ""),
                attr_data.get(REG_INFO_KEY_PARAM_TYPE, ""),
                attr_data.get(REG_INFO_KEY_TYPE, ""),
                attr_data.get(REG_INFO_KEY_VALUE, "")
            )

        # Configure data types and formats
        for dtype_format_index in range(len(inputs_types[0])):
            op_dtypes_formats = []
            # Process input data types and formats
            for input_index, _ in enumerate(inputs_types):
                dtype = inputs_types[input_index][dtype_format_index]
                format_str = inputs_formats[input_index][dtype_format_index] if dtype_format_index < len(
                    inputs_formats[input_index]) else "DefaultFormat"
                dtype_format = CustomInfoGenerator._get_dtype_format(dtype, format_str)
                op_dtypes_formats.append(dtype_format)

            # Process output data types and formats
            for output_index, _ in enumerate(outputs_types):
                dtype = outputs_types[output_index][dtype_format_index]
                format_str = outputs_formats[output_index][dtype_format_index] if dtype_format_index < len(
                    outputs_formats[output_index]) else "DefaultFormat"
                dtype_format = CustomInfoGenerator._get_dtype_format(dtype, format_str)
                op_dtypes_formats.append(dtype_format)

            custom_reg_op.dtype_format(*op_dtypes_formats)

        return custom_reg_op.target("Ascend").get_op_info()
