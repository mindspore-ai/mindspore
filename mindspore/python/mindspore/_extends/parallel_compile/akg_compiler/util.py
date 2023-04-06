# Copyright 2022 Huawei Technologies Co., Ltd
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
"""util"""
import os
import json
import shutil
from mindspore import log as logger
from mindspore._extends.parallel_compile.akg_compiler.tbe_topi import get_op_reg_info


def update_attr(attr, new_attr):
    """Update new_attr to attr."""
    if attr is None:
        attr = {}
    elif attr is str:
        attr = json.loads(attr)
    if isinstance(attr, dict):
        attr.update(new_attr)
        return json.dumps(attr)
    return attr


def get_log_level(attrs):
    """Get log level from attr."""
    attrs_dict = {}
    if isinstance(attrs, str):
        attrs_dict = json.loads(attrs)
    elif isinstance(attrs, dict):
        attrs_dict = attrs
    return attrs_dict.get("log_level")


def print_compile_log(compile_log):
    """Print compile log."""
    # compile_log format: {"AKG": {"INFO": [log1, ...]}, "TBE": {"INFO": [log1, ...]}}
    token = "=" * 20
    for compile_backend, logs in compile_log.items():
        logger.info(f"{token} {compile_backend} compile log {token}")
        for log_level, log_list in logs.items():
            for log in log_list:
                if not log:
                    continue
                if log_level == "ERROR":
                    logger.error(log)
                else:
                    logger.info(log)


def get_kernel_meta_parent_dir(attrs):
    """Get kernel_meta parent dir."""
    attrs_dict = {}
    if isinstance(attrs, str):
        attrs_dict = json.loads(attrs)
    elif isinstance(attrs, dict):
        attrs_dict = attrs
    return os.path.realpath(attrs_dict.get("compile_cache"))


def get_ascend_compile_dirs(attrs):
    """Get several Ascend compile dirs."""
    kernel_meta_dir = os.path.join(get_kernel_meta_parent_dir(attrs), "akg_kernel_meta")
    compile_dirs = {"kernel_meta_dir": kernel_meta_dir,
                    "akg_compile_dir": os.path.join(kernel_meta_dir, "akg"),
                    "tbe_compile_dir": os.path.join(kernel_meta_dir, "tbe"),
                    "composite_graph_dir": os.path.join(kernel_meta_dir, "composite")}
    return compile_dirs


def create_compile_dirs(compile_dirs):
    """Create dirs."""
    for _, d in compile_dirs.items():
        if not os.path.isdir(d):
            try:
                os.makedirs(d)
            except OSError as err:
                # File exists
                if err.errno == 17:
                    pass
                else:
                    raise err


def select_best(src_dirs, dst_dir, op_name):
    """Select best compile result."""

    def _copy_file(src_path, dst_path):
        try:
            if os.path.isfile(dst_path):
                os.remove(dst_path)
        except OSError:
            pass

        try:
            shutil.copy(src_path, dst_path)
        except PermissionError:
            # If dst_path already exits and only has READ permission
            pass

    max_block_dim = 1
    max_block_dim_idx = -1
    for i, src_dir in enumerate(src_dirs):
        o_path = os.path.join(src_dir, op_name + ".o")
        json_path = os.path.join(src_dir, op_name + ".json")
        if os.path.isfile(o_path) and os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                json_str = f.read()
                json_dict = json.loads(json_str)
                if json_dict["blockDim"] >= max_block_dim:
                    max_block_dim_idx = i
                    max_block_dim = json_dict["blockDim"]
    if max_block_dim_idx >= 0:
        o_path = os.path.join(src_dirs[max_block_dim_idx], op_name + ".o")
        json_path = os.path.join(src_dirs[max_block_dim_idx], op_name + ".json")
        _copy_file(o_path, os.path.join(dst_dir, op_name + ".o"))
        _copy_file(json_path, os.path.join(dst_dir, op_name + ".json"))
        logger.info("{}, best compile result dir: {}".format(op_name, src_dirs[max_block_dim_idx]))
        return True
    logger.info("{}, best compile result dir not found".format(op_name))
    return False


def check_tbe_support(json_desc):
    """Check if current json str is supported in TBE."""
    if "buffer_stitch" in json_desc:
        logger.info("TBE not supports buffer stitch")
        return False

    if "parallel_fusion" in json_desc:
        logger.info("TBE not supports parallel fusion")
        return False

    if not json_desc.get("input_desc"):
        logger.info("TBE not supports empty inputs")
        return False

    for op in json_desc["op_desc"]:
        op_name = op["name"]
        if not get_op_reg_info(op_name, "func", False):
            logger.info("TBE op not registered: {}".format(op_name))
            return False
    return True
