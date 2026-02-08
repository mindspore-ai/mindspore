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
""" unit management base module. """
import os
import re
import numpy as np
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore import log as logger
from mindspore import mint


def get_discontinuous_tensor(low=0, high=1, shape=(4, 9), dim=(1, 0), dtype=mstype.float32):
    x = Tensor(np.random.uniform(low, high, size=shape), dtype)
    output = mint.permute(x, dim)
    return output


def get_empty_tensor(dtype=mstype.float32, shape=(2, 0)):
    x = Tensor(np.random.randn(*shape), dtype)
    return x


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    # ICKTGQ
    if data_expected.dtype in ('complex64', 'complex128'):
        greater = greater + nan_diff + inf_diff
    else:
        neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
        greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ckpt file error.".format(e))


def find_newest_ckpt_file(folder_path, file_format="ckpt"):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith(f'.{file_format}'),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def find_newest_ckpt_file_by_name(folder_path, file_format="ckpt"):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith(f'.{file_format}'),
                            os.listdir(folder_path)))
    return max(list(ckpt_files))


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ir/dot/dat file error.".format(e))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def find_newest_begin_ir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'step_parallel+_begin_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)
