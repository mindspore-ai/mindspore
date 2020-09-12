# Copyright 2019 Huawei Technologies Co., Ltd
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
"""Validate the input path."""
import os


def validate_and_normalize_path(
        path,
        check_absolute_path=False,
        allow_parent_dir=True,
):
    """
    Validates path and returns its normalized form.

    If path has a valid scheme, treat path as url, otherwise consider path a
    unix local path.

    Note:
        File scheme (rfc8089) is currently not supported.

    Args:
        path (str): Path to be normalized.
        check_absolute_path (bool): Whether check path scheme is supported.
        allow_parent_dir (bool): Whether allow parent dir in path.

    Returns:
        str, normalized path.
    """
    if not path:
        raise RuntimeError("The path is invalid!")

    path_str = str(path)
    if not allow_parent_dir:
        path_components = path_str.split("/")
        if ".." in path_components:
            raise RuntimeError("The parent path is not allowed!")

    # path does not have valid schema, treat it as unix local path.
    if check_absolute_path:
        if not path_str.startswith("/"):
            raise RuntimeError("The path is invalid!")
    try:
        # most unix systems allow
        normalized_path = os.path.realpath(path)
    except ValueError:
        raise RuntimeError("The path is invalid!")

    return normalized_path
