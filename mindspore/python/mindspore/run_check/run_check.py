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

"""
mindspore.run_check

The goal is to provide a convenient API to check if the installation is successful or failed.
"""


def _check_mul():
    """
    Define the mul method.
    """
    from importlib import import_module
    import numpy as np

    try:
        ms = import_module("mindspore")
    except ModuleNotFoundError:
        ms = None
    finally:
        pass

    print(f"MindSpore version: ", ms.__version__)

    input_x = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    input_y = ms.Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
    mul = ms.ops.Mul()
    mul(input_x, input_y)
    print("The result of multiplication calculation is correct, MindSpore has been installed on platform "
          f"[{ms.get_context('device_target')}] successfully!")


def run_check():
    """
    Provide a convenient API to check if the installation is successful or failed.
    If there is no return value, the verification status will be displayed directly.

    Examples:
        >>> import mindspore
        >>> mindspore.run_check()
        MindSpore version: xxx
        The result of multiplication calculation is correct, MindSpore has been installed successfully!
    """
    try:
        _check_mul()
    # pylint: disable=broad-except
    except Exception as e:
        print("MindSpore running check failed.")
        print(str(e))
    finally:
        pass
