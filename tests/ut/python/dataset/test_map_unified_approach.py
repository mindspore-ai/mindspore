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
c_op2 = vision.VerticalFlip()
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
COMPOSE_NAME = "Compose"
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


def test_unified_api_post_decode():
    """
    Feature: Transforms Unification
    Description: Test the approach for ops selected with unified ops, following Decode op in Map
    Expectation: List of runtime ops are verified
    """
    # Test ops following Decode(), namely Decode(to_pil=False)
    # Decode()   Mix
    # Decode()   c
    # Decode()   pyfunc
    # Decode()   py
    validate_ops(operations_config1([decode_op, vision.Equalize()]), expected=[DECODE_OP_NAME, MIX_OP_EQUALIZE_NAME])
    validate_ops(operations_config1([decode_op, c_op]), expected=[DECODE_OP_NAME, C_OP_NAME])
    validate_ops(operations_config1([decode_op, pyfunc]), expected=[DECODE_OP_NAME, PYFUNC_NAME])
    validate_ops(operations_config1([decode_op, vision.ToPIL(), py_op]), expected=[DECODE_OP_NAME, COMPOSE_NAME])

    # Test ops following Decode(to_pil=True)
    # Decode(to_pil=True)   Mix
    # Decode(to_pil=True)   c
    # Decode(to_pil=True)   pyfunc
    # Decode(to_pil=True)   py
    validate_ops(operations_config1([decode_op_to_pil, vision.Invert()]), expected=[COMPOSE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, vision.ToPIL(), c_op]), expected=[COMPOSE_NAME, C_OP_NAME])
    validate_ops(operations_config1([decode_op_to_pil, pyfunc]), expected=[COMPOSE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, py_op]), expected=[COMPOSE_NAME])


def test_unified_api_one_map():
    """
    Feature: Transforms Unification
    Description: Test the approach for ops selected with unified ops in single map with Decode as first op in map
    Expectation: List of runtime ops are verified
    """
    # Test various combinations of ops, based on first op and subsequent ops

    # Beginning with Decode(), test ops following an op with both C++ implementation and Python implementation
    # Mix   Mix
    # Mix   c
    # Mix   pyfunc
    # Mix   py
    validate_ops(operations_config1([decode_op, vision.Equalize(), vision.Invert()]),
                 expected=[DECODE_OP_NAME, MIX_OP_EQUALIZE_NAME, MIX_OP_INVERT_NAME2])
    validate_ops(operations_config1([decode_op, vision.Equalize(), c_op]),
                 expected=[DECODE_OP_NAME, MIX_OP_EQUALIZE_NAME, C_OP_NAME])
    validate_ops(operations_config1([decode_op, vision.Equalize(), pyfunc]),
                 expected=[DECODE_OP_NAME, MIX_OP_EQUALIZE_NAME, PYFUNC_NAME])
    validate_ops(operations_config1([decode_op, vision.Equalize(), vision.ToPIL(), py_op]),
                 expected=[DECODE_OP_NAME, MIX_OP_EQUALIZE_NAME, COMPOSE_NAME])

    # Beginning with Decode(to_pil=True), test ops following an op with both C++ implementation and
    #     Python implementation
    # Mix   Mix
    # Mix   c
    # Mix   pyfunc
    # Mix   py
    validate_ops(operations_config1([decode_op_to_pil, vision.AdjustGamma(10.0), vision.Invert()]),
                 expected=[COMPOSE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, vision.AdjustGamma(10.0), c_op]),
                 expected=[COMPOSE_NAME, C_OP_NAME])
    validate_ops(operations_config1([decode_op_to_pil, vision.AdjustGamma(10.0), vision.ToNumpy(), c_op]),
                 expected=[COMPOSE_NAME, TONUMPY_NAME, C_OP_NAME])
    validate_ops(operations_config1([decode_op_to_pil, vision.AdjustGamma(10.0), pyfunc]), expected=[COMPOSE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, vision.AdjustGamma(10.0), py_op]), expected=[COMPOSE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, vision.AdjustGamma(10.0), vision.ToPIL(), py_op]),
                 expected=[COMPOSE_NAME])

    # Beginning with Decode(), test ops following C++ op
    # c   Mix
    # c   c
    # c   pyfunc
    # c   py
    validate_ops(operations_config1([decode_op, c_op2, vision.Equalize()]),
                 expected=[DECODE_OP_NAME, C_OP_NAME2, MIX_OP_EQUALIZE_NAME])
    validate_ops(operations_config1([decode_op, c_op, c_op2]), expected=[DECODE_OP_NAME, C_OP_NAME, C_OP_NAME2])
    validate_ops(operations_config1([decode_op, c_op, pyfunc]), expected=[DECODE_OP_NAME, C_OP_NAME, PYFUNC_NAME])
    validate_ops(operations_config1([decode_op, c_op2, vision.ToPIL(), py_op]),
                 expected=[DECODE_OP_NAME, C_OP_NAME2, COMPOSE_NAME])

    # Beginning with Decode(), test ops following pyfunc
    # pyfunc   Mix
    # pyfunc   c
    # pyfunc   pyfunc
    # pyfunc   py
    validate_ops(operations_config1([decode_op, vision.ToPIL(), pyfunc, vision.Equalize()]),
                 expected=[DECODE_OP_NAME, COMPOSE_NAME, MIX_OP_EQUALIZE_NAME])
    validate_ops(operations_config1([decode_op, vision.ToPIL(), pyfunc, c_op]),
                 expected=[DECODE_OP_NAME, COMPOSE_NAME, C_OP_NAME])
    validate_ops(operations_config1([decode_op, vision.ToPIL(), pyfunc, pyfunc2]),
                 expected=[DECODE_OP_NAME, COMPOSE_NAME])
    validate_ops(operations_config1([decode_op, vision.ToPIL(), pyfunc, vision.ToPIL(), py_op]),
                 expected=[DECODE_OP_NAME, COMPOSE_NAME])

    # Beginning with Decode(to_pil=True), test ops following pyfunc
    # pyfunc   Mix
    # pyfunc   c
    # pyfunc   pyfunc
    # pyfunc   py
    validate_ops(operations_config1([decode_op_to_pil, pyfunc, vision.Equalize()]),
                 expected=[COMPOSE_NAME, MIX_OP_EQUALIZE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, pyfunc, c_op]), expected=[COMPOSE_NAME, C_OP_NAME])
    validate_ops(operations_config1([decode_op_to_pil, pyfunc, pyfunc2]), expected=[COMPOSE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, pyfunc, vision.ToPIL(), py_op]), expected=[COMPOSE_NAME])

    # Beginning with Decode(to_pil=True), test ops following Python op
    # py   Mix
    # py   c
    # py   pyfunc
    # py   py
    validate_ops(operations_config1([decode_op_to_pil, py_op, vision.Equalize()]), expected=[COMPOSE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, py_op, vision.ToPIL(), c_op]),
                 expected=[COMPOSE_NAME, C_OP_NAME])
    validate_ops(operations_config1([decode_op_to_pil, py_op, pyfunc]), expected=[COMPOSE_NAME])
    validate_ops(operations_config1([decode_op_to_pil, py_op, py_op2]), expected=[COMPOSE_NAME])


def test_unified_api():
    """
    Feature: Transforms Unification
    Description: Test the approach for ops selected with unified ops, with decoded data (namely Decode in prior Map)
    Expectation: List of runtime ops are verified
    """
    # Test various combinations of ops, based on first op and subsequent ops

    # Test ops following an op with both C++ implementation and Python implementation
    # Mix   Mix
    # Mix   c
    # Mix   pyfunc
    # Mix   py
    validate_ops(operations_config2([vision.Equalize(), vision.Invert()], is_pil=False),
                 expected=[MIX_OP_EQUALIZE_NAME, MIX_OP_INVERT_NAME2])
    validate_ops(operations_config2([vision.Equalize(), c_op], is_pil=False),
                 expected=[MIX_OP_EQUALIZE_NAME, C_OP_NAME])
    validate_ops(operations_config2([vision.Equalize(), pyfunc], is_pil=False),
                 expected=[MIX_OP_EQUALIZE_NAME, PYFUNC_NAME])
    validate_ops(operations_config2([vision.Equalize(), vision.ToPIL(), py_op], is_pil=False),
                 expected=[MIX_OP_EQUALIZE_NAME, COMPOSE_NAME])

    # Test ops following C++ op
    # c   Mix
    # c   c
    # c   pyfunc
    # c   py
    validate_ops(operations_config2([c_op2, vision.Equalize()], is_pil=False),
                 expected=[C_OP_NAME2, MIX_OP_EQUALIZE_NAME])
    validate_ops(operations_config2([c_op, c_op2], is_pil=False), expected=[C_OP_NAME, C_OP_NAME2])
    validate_ops(operations_config2([c_op, pyfunc], is_pil=False), expected=[C_OP_NAME, PYFUNC_NAME])
    validate_ops(operations_config2([c_op2, vision.ToPIL(), py_op], is_pil=False), expected=[C_OP_NAME2, COMPOSE_NAME])

    # Test ops following pyfunc
    # pyfunc   Mix
    # pyfunc   c
    # pyfunc   pyfunc
    # pyfunc   py
    validate_ops(operations_config2([pyfunc, vision.Equalize()], is_pil=False),
                 expected=[PYFUNC_NAME, MIX_OP_EQUALIZE_NAME])
    validate_ops(operations_config2([pyfunc, c_op], is_pil=False), expected=[PYFUNC_NAME, C_OP_NAME])
    validate_ops(operations_config2([pyfunc, pyfunc2], is_pil=False), expected=[COMPOSE_NAME])
    validate_ops(operations_config2([pyfunc, vision.ToPIL(), py_op], is_pil=False), expected=[COMPOSE_NAME])

    # Test ops following Python op
    # py   Mix
    # py   c
    # py   pyfunc
    # py   py
    validate_ops(operations_config2([vision.ToPIL(), py_op, vision.Equalize()], is_pil=True), expected=[COMPOSE_NAME])
    validate_ops(operations_config2([vision.ToPIL(), py_op, vision.ToNumpy(), c_op], is_pil=True),
                 expected=[COMPOSE_NAME, TONUMPY_NAME, C_OP_NAME])
    validate_ops(operations_config2([vision.ToPIL(), py_op, pyfunc], is_pil=True), expected=[COMPOSE_NAME])
    validate_ops(operations_config2([vision.ToPIL(), py_op, py_op2], is_pil=True), expected=[COMPOSE_NAME])


if __name__ == "__main__":
    test_unified_api_post_decode()
    test_unified_api_one_map()
    test_unified_api()
