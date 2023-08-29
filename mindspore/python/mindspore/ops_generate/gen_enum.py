# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
Generate enum definition from enum.yaml
"""
import os
import yaml

def generate_py_enum(yaml_data):
    """
    generate python enum
    """
    gen_py = ''
    blank_line = f"""
"""
    for enum_name, enum_data in yaml_data.items():
        class_name = ''.join(word.capitalize() for word in enum_name.split('_'))
        enum_keys = []
        enum_values = []
        enum_class_code = f"""
class {class_name}(Enum):
"""
        enum_func_code = f"""
def {enum_name}_to_enum({enum_name}_str):
"""
        for enum_key, enum_value in enum_data.items():
            enum_keys.append(enum_key)
            enum_values.append(enum_value)
            enum_class_code += f"""    {enum_key} = {enum_value}
"""
            enum_func_code += f"""    if {enum_name}_str == "{enum_key}":
        return {enum_value}
"""

        value_error_code = f"""    raise ValueError(f"Invalid {class_name}: {{{enum_name}_str}}")
"""
        gen_py += enum_class_code + blank_line + enum_func_code + value_error_code + blank_line
    return gen_py


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    work_path = os.path.join(current_path, '../../../../')
    yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/enum.yaml')
    enum_py_path = os.path.join(work_path, 'mindspore/python/mindspore/ops/auto_generate/gen_enum_def.py')

    yaml_str = None
    with open(yaml_path, 'r') as yaml_file:
        yaml_str = yaml.safe_load(yaml_file)

    py_licence_str = f"""# Copyright 2023 Huawei Technologies Co., Ltd
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

    pyheader = f"""
\"\"\"Operator argument enum definition.\"\"\"

from enum import Enum

"""
    py_enum = generate_py_enum(yaml_str)
    with open(enum_py_path, 'w') as py_file:
        py_file.write(py_licence_str + pyheader + py_enum)
