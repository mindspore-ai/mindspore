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
"""mindir utility."""
from __future__ import absolute_import

import os
from mindspore import log as logger
from mindspore import _checkparam as Validator
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model


def load_mindir(file_name):
    """
    load protobuf file.

    Args:
        file_name (str): File name.

    Returns:
        ModelProto, mindir proto object.

    Raises:
        ValueError: The file does not exist or the file name format is incorrect.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> md = ms.load_mindir("test.mindir")
    """

    Validator.check_file_name_by_regular(file_name)
    file_name = os.path.realpath(file_name)
    model = mindir_model()

    try:
        with open(file_name, "rb") as f:
            pb_content = f.read()
            model.ParseFromString(pb_content)
    except BaseException as e:
        logger.critical(f"Failed to parse the file: {file_name} "
                        f" please check the correct file.")
        raise ValueError(e.__str__()) from e
    finally:
        pass

    return model


def save_mindir(model, file_name):
    """
    save protobuf file.

    Args:
        model (ModelProto): mindir model
        file_name (str): File name.

    Raises:
        TypeError: The argument `model` is not a ModelProto object.
        ValueError: The file path does not exist or the `file_name` format is incorrect.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> md = ms.load_mindir("test.mindir")
        >>> md.user_info["version"]="pangu v100"
        >>> ms.save_mindir(md,"test_new.mindir")
        >>> md_new = ms.load_mindir("test_new.mindir")
        >>> md_new.user_info
    """

    Validator.check_file_name_by_regular(file_name)
    file_name = os.path.realpath(file_name)

    if not isinstance(model, mindir_model):
        raise TypeError("For 'save_mindir', the argument 'model' must be ModelProto, "
                        "but got {}.".format(type(model)))
    try:
        with open(file_name, "wb") as f:
            f.write(model.SerializeToString())
    except BaseException as e:
        logger.critical(f"Failed to save the file: {file_name} ,"
                        f" please check the correct file.")
        raise ValueError(e.__str__()) from e
    finally:
        pass
