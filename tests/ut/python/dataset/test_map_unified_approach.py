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
# ==============================================================================
"""
Test Unified API Approach to op implementation selection for Map op
"""
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

DATA_DIR = "../data/dataset/testImageNetData/train/"

# Decode op constants and variables
decode_op = vision.Decode()
DECODE_OP_NAME = "DecodeOperation"
decode_op_to_pil = vision.Decode(to_pil=True)

# List of C++ implementation-only ops
c_op = vision.Rescale(0.5, 0)
C_OP_NAME = "RescaleOperation"
C_OP_NAME2 = "VerticalFlipOperation"

# List of Python implementation-only ops
py_op = vision.Grayscale(3)
py_op2 = vision.RandomPerspective(0.4, 1.0)

# List of mixed ops with both C++ implementation and Python implementation
# Note: Global constants are not used for the mixed ops to avoid tests having any code dependency that code
#       may or may not reuse TensorOperation settings like self.implementation
MIX_OP_EQUALIZE_NAME = "EqualizeOperation"
MIX_OP_INVERT_NAME2 = "InvertOperation"
MIX_OP_ADJUSTGAMMA_NAME3 = "AdjustGammaOperation"

# Other op constants
PYFUNC_NAME = "FuncWrapper"
COMPOSE_NAME = "_Compose"
TONUMPY_NAME = "ToNumpy"


def pyfunc(x):
    """Simple Python function"""
    return x


def pyfunc2(y):
    """Simple Python function"""
    return y


def operations_config1(transforms_list):
    """ For dataset pipeline with single map, return the dataset operations. """
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False, num_samples=1)
    data = data.map(transforms_list, num_parallel_workers=2, python_multiprocessing=False)

    itr = data.create_dict_iterator(num_epochs=1, output_numpy=True)
    operations = None
    for _ in itr:
        if not operations:
            operations = itr.dataset.operations
    return operations


def operations_config2(transforms_list, is_pil):
    """ For dataset pipeline with map#1 with Decode, and map#2 with transforms list, return the dataset operations. """
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False, num_samples=2)
    data = data.map(vision.Decode(is_pil), num_parallel_workers=2, python_multiprocessing=False)
    data = data.map(transforms_list, num_parallel_workers=2, python_multiprocessing=False)

    itr = data.create_dict_iterator(num_epochs=1, output_numpy=True)
    operations = None
    for _ in itr:
        if not operations:
            operations = itr.dataset.operations
    return operations


def validate_ops(actual, expected):
    """ Validate the ops executed in the actual runtime tree match the expected list of ops. """
    assert all(a.__class__.__name__ == e for a, e in zip(actual, expected))
