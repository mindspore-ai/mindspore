# Copyright 2021 Huawei Technologies Co., Ltd
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
"""HPC generator"""

import sys
import os
import io
import argparse
from itertools import chain

def key_value_pair(line):
    """
    split key and value
    :param line:
    :return:
    """
    key = None
    value = None
    try:
        key, value = line.split("=", 1)
    except ValueError:
        print("line must be format: key=value, but now is:", line)
        sys.exit(1)
    try:
        value = int(value)
    except ValueError:
        print("Error: you input value must be integer, but now is:", value)
        sys.exit(1)
    return key, value

def get_indent(line):
    """
    get indent length
    :param line:
    :return:
    """
    index = 0
    for i in line:
        if i == " ":
            index += 1
        else:
            break
    return index

def print_line(line):
    """
    Convert line to a python string
    :param line:
    :return:
    """
    global PYTHON_INDENT
    global GENERATE_CODE_INDENT
    if line.strip()[0] == "}" or line.strip()[0] == ")":
        PYTHON_INDENT = -1
    split_str = line.split("@")
    if line.strip()[0] != "@" and len(split_str) == 1:
        if get_indent(line) == PYTHON_INDENT or PYTHON_INDENT == -1:
            result = ["print(", line, ", file=OUT_STREAM)"]
            PYTHON_INDENT = -1
            if "{" in line or "asm volatile(" in line:
                GENERATE_CODE_INDENT = get_indent(line)
            if line.strip().startswith("}") and "{" not in line:
                GENERATE_CODE_INDENT -= 4
            if len(line) == 1 and line[0] == "}":
                # modify next fun GENERATE_CODE_INDENT
                GENERATE_CODE_INDENT = -4
            return "\"".join(result)

    if line.strip()[0] == '@':
        # get python indent and first GENERATE_CODE_INDENT
        if PYTHON_INDENT == -1:
            GENERATE_CODE_INDENT = get_indent(line) - 4
            PYTHON_INDENT = get_indent(line)
        result = split_str[0][PYTHON_INDENT:] + split_str[1]
        return result

    index = get_indent(split_str[0])
    result = [split_str[0][PYTHON_INDENT:index] + "print("]
    prefix = " " * (GENERATE_CODE_INDENT + 4) + split_str[0].lstrip()

    suffix = " %("
    for str_tmp in split_str[1:]:
        second = str_tmp.find("}")
        suffix += str_tmp[1:second] + ', '
        str_tmp = str_tmp.replace(str_tmp[0:second + 1], "%d")
        prefix += str_tmp
    result.append(prefix)
    result.append(suffix + "), file=OUT_STREAM)")
    return "\"".join(result)

def generate_code(template_file, exec_dict):
    """
    generate hpc
    :param template_file: template file path
    :param exec_dict: dict
    :return: hpc
    """
    output_stream = io.StringIO()
    with open(template_file, 'r') as f:
        generate_code_lines = []
        for line in f:
            line = line.replace("\n", "")
            if line.strip() and line.strip()[0] != "@":
                line = line.replace("\"", "\\\"")
                line = line.replace("%", "%%")
            if "print" in line:
                line = line.replace("%%", "%")
            if not line:
                generate_code_lines.append("print(" + "\"" + line + "\"" + ", file=OUT_STREAM)")
            else:
                str = print_line(line)
                if "%(" not in str:
                    str = str.replace("%%[", "%[")
                generate_code_lines.append(str)
        c = compile('\n'.join(generate_code_lines), '', 'exec')
        exec_dict["OUT_STREAM"] = output_stream
        exec(c, exec_dict)
    return output_stream.getvalue()

def check_python_version():
    if sys.version_info < (3, 6):
        sys.stdout.write("At least python 3.6 is required, but now is " + str(sys.version_info.major) + "." +
                         str(sys.version_info.minor) + "\n")
        sys.exit(1)

GENERATE_CODE_INDENT = -4
PYTHON_INDENT = -1

parser = argparse.ArgumentParser(description="MSLite NNACL Code Generator")
parser.add_argument("-I", dest="Template_File", nargs=1, help="template file to generate code")
parser.add_argument("-A", dest="defines", metavar="KEY=VALUE", nargs="*", type=key_value_pair, action="append",
                    help="Custom Parameters")
parser.add_argument("-O", dest="Output_File", nargs=1, help="generate code output file path")

if __name__ == "__main__":
    check_python_version()
    parameters = parser.parse_args(sys.argv[1:])
    exec_globals = dict(chain(*parameters.defines))

    generate_code_str = generate_code(parameters.Template_File[0], exec_globals)
    if os.path.exists(parameters.Output_File[0]):
        os.remove(parameters.Output_File[0])

    saveDir = os.path.dirname(parameters.Output_File[0])
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    with open(parameters.Output_File[0], "w", encoding='utf-8') as output_file:
        output_file.write(generate_code_str)
