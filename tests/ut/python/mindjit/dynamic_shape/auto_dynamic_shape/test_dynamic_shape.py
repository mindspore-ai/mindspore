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

import os
import subprocess


def generate_dyn(file_name, dyn_file_name):
    if os.path.exists(dyn_file_name):
        os.remove(dyn_file_name)
    assert not os.path.exists(dyn_file_name)

    dirname = os.path.dirname(os.path.abspath(__file__))
    cmd = f"VLOG_v=1 python " + dirname + "/" + file_name + " > " + dyn_file_name + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(dyn_file_name)
    with open(dyn_file_name, "r") as v_file:
        data = v_file.read()

    assert data.count("Start compiling") == 3
    assert data.count("End compiling") == 3

    # Clean files
    os.remove(dyn_file_name)


def test_dynamic_shape_with_input_tensor():
    """
    Feature: Add dynamic shape feature.
    Description: The static shape is automatically converted to a dynamic shape.
    Expectation: The compile number is 3.
    """
    generate_dyn("dynamic_shape_input_tensor.py", "dynamic_shape_vlog1.log")


def test_dynamic_shape_with_input_tuple_tensor():
    """
    Feature: Add dynamic shape feature.
    Description: The static shape is automatically converted to a dynamic shape.
    Expectation: The compile number is 3.
    """
    generate_dyn("dynamic_shape_input_tuple_tensor.py", "dynamic_shape_vlog2.log")


def test_dynamic_shape_with_input_float_tensor():
    """
    Feature: Add dynamic shape feature.
    Description: The static shape is automatically converted to a dynamic shape.
    Expectation: The compile number is 3.
    """
    generate_dyn("dynamic_shape_input_float.py", "dynamic_shape_vlog3.log")
