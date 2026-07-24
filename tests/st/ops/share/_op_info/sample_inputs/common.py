# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
# pylint: skip-file
"""
Common functions for operation testing.
This module provides common sample input builders for various ops, including
basic, extra, dynamic, error case generators and small combination tools.
"""
import itertools
import mindspore as ms
from tests.st.ops.share._op_info.op_info import OpInfo

from tests.st.ops.share._internal.utils import (
    OpSampleInput, make_tensor,
)
from tests.st.ops.share._op_info.op_common import (
    dtypes_extra_values
)


def _small_large_values(dtype):
    values = []
    if dtype == ms.uint8:
        values = dtypes_extra_values['_uint_small_values']
    elif dtype in [ms.int8, ms.int16, ms.int32, ms.int64]:
        values = dtypes_extra_values['_int_small_values'] + dtypes_extra_values['_int_large_values']
    else:
        if dtype in [ms.float16]:
            values = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp16_large_values']
        else:
            float_vals = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp32_large_values']
            if dtype.is_complex:
                values = [complex(r, i) for r, i in itertools.product(float_vals, float_vals)]
            else:
                values = float_vals
    return values

def _generate_binary_inplace_op_broadcasting_and_discontiguous_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate broadcasting and (dis)contiguous tensor inputs for in-place operations.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering broadcasting and contiguous/non-contiguous memory layouts.
    """

    shapes = (
        ((1,), ()),
        ((3,), ()),
        ((3,), (1,)),
        ((3, 1), (1,)),
        ((1, 3), (3,)),
        ((3, 2), (2,)),
        ((1, 2, 3), (3,)),
        ((1, 2, 3), (2, 3)),
        ((2, 1, 3), (1, 3)),
        ((2, 3, 4), ()),
        ((2, 1, 2), (1, 1, 2)),
    )

    for shape_pair, discontiguous in itertools.product(shapes, [False, True]):
        _input_shape, _other_shape = shape_pair

        yield OpSampleInput(
            op_input=make_tensor(
                _input_shape,
                dtype=dtype,
                device=device,
                discontiguous=discontiguous,
                low=op_info.input_low,
                high=op_info.input_high,
                exclude_value=op_info.input_exclude_value,
            ),
            op_args=(
                make_tensor(
                    _other_shape,
                    dtype=dtype,
                    device=device,
                    discontiguous=discontiguous,
                    low=op_info.other_low,
                    high=op_info.other_high,
                    exclude_value=op_info.other_exclude_value,
                ),
            ),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_{len(_input_shape)}d_x"+
                        f"_{len(_other_shape)}d_broadcasting_and_discontiguous_{discontiguous}_tensor_inputs",
        )
