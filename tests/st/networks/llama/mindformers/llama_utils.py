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
"""Trainer Utils."""
import importlib.util
import os
import random
from enum import Enum
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import set_seed as ms_set_seed
from mindspore.communication import get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode


str_to_ms_type = {
    "float16": mstype.float16,
    "float32": mstype.float32,
    "bfloat16": mstype.bfloat16
}

class BaseEnum(str, Enum):
    """
    Base Enum for MindFormers.
    """

    @classmethod
    def _missing_(cls, value):
        """Enum with more explicit error message for missing values."""
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class SaveIntervalStrategy(BaseEnum):
    """
    Stores the acceptable string identifiers for save checkpoint monitor.
    """
    NO = "no"
    STEPS = "steps"
    SECONDS = "seconds"


class LRType(BaseEnum):
    """
    Stores the acceptable string identifiers for learning rate schedule.
    """
    # will be support item for future.
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class OptimizerType(BaseEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    # supported item for test, will be delete in the future.
    ADAMWEIGHTDECAY = 'AdamWeightDecay'

    # will be support item for future.
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAFACTOR = "adafactor"


class WrapperType(BaseEnum):
    """
    Stores the acceptable string identifiers for training wrapper.
    """
    # will be support item for future.
    MFWRAPPER = 'mf_wrapper'
    TRAINONESTEP = 'wrapper'
    TRAINONESTEPWITHLOSSSCALE = 'loss_scale_wrapper'


def set_seed(seed: int = 0):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `MindSpore`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    ms_set_seed(seed)


def check_keywords_in_name(name, keywords=()):
    """ Check keywords in name. """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def str2bool(b):
    """String convert to Bool."""
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def check_runner_config(config, dataset):
    """ Check runner config. """
    data_size = dataset.get_dataset_size()
    new_epochs = config.runner_config.epochs
    config.runner_config.origin_epochs = new_epochs
    if config.runner_config.gradient_accumulation_steps is None:
        config.runner_config.gradient_accumulation_steps = 1
    if config.runner_config.initial_epoch is None:
        config.runner_config.initial_epoch = 0
    if config.runner_config.sink_mode:
        if config.runner_config.sink_size != -1:
            if config.runner_config.sink_size <= 0:
                raise ValueError("per epoch size must be more than 0 or equal to -1, "
                                 f"but get {config.runner_config.sink_size}")
            config.runner_config.epochs = int((data_size / config.runner_config.sink_size) * new_epochs)
        else:
            config.runner_config.sink_size = data_size
    else:

        config.runner_config.sink_size = -1

    config.data_size = data_size


def check_dataset_config(config):
    """Check dataset config."""
    if config.train_dataset is not None:
        config.train_dataset.do_eval = False
        config.train_dataset.seed = config.seed
        config.train_dataset.auto_tune = config.auto_tune
        config.train_dataset.filepath_prefix = config.filepath_prefix
        config.train_dataset.autotune_per_step = config.autotune_per_step
        config.train_dataset.profile = config.profile
        config.train_dataset.batch_size = config.runner_config.batch_size
        if config.train_dataset.mixup_op:
            config.train_dataset.mixup_op.num_classes = config.runner_config.num_classes
        config.train_dataset_task.dataset_config = config.train_dataset


def get_real_rank():
    try:
        return get_rank()
    except RuntimeError:
        return int(os.getenv("RANK_ID", "0"))


def get_real_group_size():
    try:
        return get_group_size()
    except RuntimeError:
        return int(os.getenv("RANK_SIZE", "1"))


def get_dataset_map(dataset, operations, input_columns=None, output_columns=None, num_parallel_workers=None, **kwargs):
    return dataset.map(operations,
                       input_columns=input_columns,
                       output_columns=output_columns,
                       num_parallel_workers=num_parallel_workers,
                       **kwargs)


def replace_tk_to_mindpet(ckpt_dict):
    """replace 'tk_delta' in para name to 'mindpet_delta'"""
    ckpt_new = {}
    for k, v in ckpt_dict.items():
        ckpt_new[k.replace('tk_delta', 'mindpet_delta')] = v
    return ckpt_new


PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


def convert_mstype(ms_type: str = "float16"):
    """Convert the string type to MindSpore type."""
    if isinstance(ms_type, mstype.Float):
        return ms_type
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "float32":
        return mstype.float32
    if ms_type == "bfloat16":
        return mstype.bfloat16
    raise KeyError(f"Supported data type keywords include: "
                   f"[float16, float32, bfloat16], but get {ms_type}")


def check_valid_big_kernel():
    """check mindspore version is valid for big kernel SiLU and LlamaRMSNorm Ops"""
    version_valid = is_version_ge(ms.__version__, "2.2.10")
    # below ms 2.2.10 is not support
    if not version_valid:
        result = False
    else:
        result = True
    return result


def is_version_ge(current_version, base_version):
    """
        return current_version >= base_version.
        Check whether the current version is higher than or equal to the base version.
        for current_version: 1.8.1, base_version: 1.11.0, it return False.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1， base_version: 1.11.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True


def get_ascend_soc_version():
    """Get ascend soc version."""
    if is_version_ge(ms.__version__, "2.2.0"):
        from mindspore._c_expression import MSContext
        return MSContext.get_instance().get_ascend_soc_version()
    ascend_chip_type = os.getenv("ASCEND_CHIP_TYPE", "UNSET")
    if ascend_chip_type not in ["910a", "910b", "UNSET"]:
        raise EnvironmentError(f"ASCEND_CHIP_TYPE should be in ['910a', '910b'],but get {ascend_chip_type}")
    return ascend_chip_type


def is_910a():
    device = get_ascend_soc_version()
    return device in ['910a', 'ascend910']


def is_910b():
    device = get_ascend_soc_version()
    return device in ['910b', 'ascend910b']


def check_rmsnorm_big_kernel_valid(is_dynamic=False):
    """check whether rmsnorm big kernel is valid"""
    if check_valid_big_kernel() and not is_910a() and not is_dynamic:
        return True
    return False


def check_valid_paged_attention():
    """check mindspore version is valid for paged attention"""
    version_valid = is_version_ge(ms.__version__, "2.2.11")
    # below ms 2.2.11 is not support
    if not version_valid:
        result = False
    else:
        result = True
    return result


def check_valid_flash_attention(import_fa_valid=True, fa_type=None):
    """check mindspore version is valid for input flash attention"""
    version_map = {"IncreFlashAttention": "2.3.0",
                   "PromptFlashAttention": "2.2.0",
                   "FlashAttention": "2.2.0"}
    valid_version = version_map.get(fa_type)
    if not is_910b() and fa_type in ["PromptFlashAttention", "IncreFlashAttention"]:
        return False
    if valid_version is None:
        raise ValueError(f"fa_type should be in {list(version_map.keys())}, but get {fa_type}")
    version_valid = is_version_ge(ms.__version__, valid_version)
    if not version_valid:
        result = False
    elif not import_fa_valid:
        result = False
    elif fa_type == "IncreFlashAttention" and _get_parallel_mode() not in ParallelMode.STAND_ALONE:
        result = False
    # both pass should return True
    else:
        result = True
    return result


def get_cell_reuse(func):
    """Cell reuse decorator."""

    def decorator(*args, **kwargs):
        stand_alone = ms.get_auto_parallel_context("parallel_mode") == 'stand_alone'
        pipline_parallel = ms.get_auto_parallel_context("pipeline_stages") > 1
        if os.getenv("ENABLE_CELL_REUSE", "0") == "0" or \
                not is_version_ge(ms.__version__, "2.1.0") \
                or stand_alone or not pipline_parallel:
            func(*args, **kwargs)
            return
        from mindspore.common import lazy_inline
        lazy_inline(func)(*args, **kwargs)

    return decorator


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None

def reverse_dict(d: dict):
    new_d = {}
    for k, v in d.items():
        if v in new_d:
            raise ValueError(f"Different keys in dict have same values.")
        new_d[v] = k
    return new_d

ms_type_to_str = reverse_dict(str_to_ms_type)


cell_reuse = get_cell_reuse
