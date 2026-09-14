# Copyright 2026 Huawei Technologies Co., Ltd

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
# pylint: disable=ungrouped-imports
"""
Common functions for operation testing.
This module provides common sample input builders for various ops, including
basic, extra, dynamic, error case generators and small combination tools.

"""
import warnings
import itertools
import functools
import mindspore as ms
from mindspore import mint
import numpy as np
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_info import (
    _generate_unary_op_extremal_value_tensor_inputs_func,
    _generate_unary_op_large_value_tensor_inputs_func,
    _generate_unary_op_small_value_tensor_inputs_func,
    _generate_unary_op_tensors_sample_inputs_func,
)

from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array,
)
from tests.st.ops.share._op_info.op_common import (
    EXTRA_SMALL_DIM_SIZE,
    LARGE_DIM_SIZE,
    MEDIUM_DIM_SIZE,
    SMALL_DIM_SIZE,
    dtypes_extra_uint,
    dtypes_integral,
    small_large_values,
    dtypes_extra_values
)
from tests.st.ops.share._op_info.op_wrappers import Net

def basic_op_sample_inputs_dtype_grad_position_cover(basic_func, no_grad_dtype):
    def basic_op_sample_inputs(op_info,dtype, device=None, **input_kwargs):
        for sample in basic_func(op_info, dtype, device, **input_kwargs):
            if dtype == no_grad_dtype:
                sample.grad_position=()
            yield sample
    return basic_op_sample_inputs

def make_extra_tensor(op_info, op_input_shape, op_args_shapes, op_kwargs, dtype=None, device=None,
                      exclude_value=(), empty_tensor=False, dtype_promotion=False, support_multi_dim=False):

    '''
    (1, 2, 3) : need to make tensor
    ((1, 2, 3),): don't need to make tensor
    '''


    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    def _tuple_to_specified_size_tensor(data, size):
        return make_tensor_with_np_array(
            np.random.choice(np.array(data)[~np.isin(np.array(data), exclude_value)], size=size),
            dtype=dtype, device=device)

    def _whether_to_tensor(shape):
        return not (len(shape) == 1 and isinstance(shape[0], tuple))

    def _make_extra_op(data=None, op_input_shape=None, op_args_shapes=None,
                       op_kwargs=None, data_title=None,  discontiguous=False):

        def shape_convert_to_tensor(shape, discontiguous=False):
            if not _whether_to_tensor(shape):
                return shape[0]
            if not discontiguous:
                return _tuple_to_specified_size_tensor(data, shape)
            return make_arg(shape, discontiguous=True)

        if not discontiguous:
            yield OpSampleInput(
                op_input=shape_convert_to_tensor(op_input_shape),
                op_args=tuple(shape_convert_to_tensor(i) if isinstance(i,tuple) else i for i in op_args_shapes),
                op_kwargs={k: shape_convert_to_tensor(v) if isinstance(v,tuple) else v for k, v in op_kwargs.items()},
                sample_name=op_info.name + data_title,
            )
        else:
            yield OpSampleInput(
                op_input=shape_convert_to_tensor(op_input_shape, discontiguous=True),
                op_args=tuple(shape_convert_to_tensor(i, discontiguous=True) if
                              isinstance(i, tuple) else i for i in op_args_shapes),
                op_kwargs={k: shape_convert_to_tensor(v, discontiguous=True) if
                isinstance(v, tuple) else v for k, v in op_kwargs.items()},
                sample_name=f'{op_info.name}_discontiguous_{dtype}',
            )

    if empty_tensor:
        yield OpSampleInput(
            op_input=make_arg(op_input_shape),
            op_args=tuple( i[0]  if isinstance(i, tuple) and not _whether_to_tensor(i)
                           else make_arg(i) if  isinstance(i, tuple)  else i for i in op_args_shapes),
            op_kwargs={k: v[0]  if isinstance(v, tuple) and not _whether_to_tensor(v)
            else make_arg(v) if isinstance(v, tuple) else v for k, v in op_kwargs.items()},
            sample_name=f'{op_info.name}_empty_tensor_{dtype}',
        )
    elif support_multi_dim:
        yield OpSampleInput(
            op_input=make_arg(op_input_shape),
            op_args=tuple(i[0] if isinstance(i, tuple) and not _whether_to_tensor(i)
                          else make_arg(i) if isinstance(i, tuple) else i for i in op_args_shapes),
            op_kwargs={k: v[0] if isinstance(v, tuple) and not _whether_to_tensor(v)
            else make_arg(v) if isinstance(v, tuple) else v for k, v in op_kwargs.items()},
            sample_name=f'{op_info.name}_multi_dim_{dtype}',
        )
    else:
        data_dict={}
        if dtype is not None and dtype.is_floating_point:
            data_dict.update({f'{dtype}_small_values':dtypes_extra_values['_float_small_values']})
            if dtype == ms.float16:
                data_dict.update({f'{dtype}_large_values': dtypes_extra_values['_fp16_large_values']})
            else:
                data_dict.update({f'{dtype}_large_values': dtypes_extra_values['_fp32_large_values']})

            if not op_info.disable_extremal_value_tensor_inputs:
                if device != 'ascend' or MSContext.get_instance().get_ascend_soc_version() != 'ascend910':
                    data_dict.update({f'_extremal_value_np.inf_{dtype}':(np.inf,),
                                      f'_extremal_value_-np.inf_{dtype}': (-np.inf,),
                                      f'_extremal_value_np.nan_{dtype}':(np.nan,)})

        if dtype in dtypes_integral and dtype != ms.bool_:
            if dtype in dtypes_extra_uint or dtype == ms.uint8:
                data_dict.update({f'{dtype}_small_values':[dtypes_extra_values['_uint_small_values']]})
            else:
                data_dict.update({f'{dtype}_small_values':[dtypes_extra_values['_int_small_values']]})

            if dtype in [ms.int16, ms.int32, ms.int64]:
                data_dict.update({f'{dtype}_large_values': [dtypes_extra_values['_int_large_values']]})

        for data_title,data in data_dict.items():
            yield  from _make_extra_op(data, op_input_shape, op_args_shapes, op_kwargs, data_title )

        # ---- discontiguous tensors ----
        if not op_info.disable_discontiguous_tensor_inputs:
            yield from _make_extra_op(op_input_shape=op_input_shape, op_args_shapes=op_args_shapes, op_kwargs=op_kwargs,
                                      discontiguous=True,)

        if dtype_promotion:
            support_dtype = list(op_info.get_supported_dtypes(device))
            support_dtype.remove(dtype)
            for other_dtype in support_dtype:

                yield OpSampleInput(
                    op_input=make_tensor(op_input_shape, dtype=dtype, device=device),
                    op_args=tuple(i[0] if isinstance(i, tuple) and not _whether_to_tensor(i) else
                                  make_tensor(i, dtype=other_dtype, device=device)
                                  if isinstance(i, tuple) else i for i in op_args_shapes),
                    op_kwargs={k: v[0] if isinstance(v, tuple) and not _whether_to_tensor(v)
                    else make_tensor(v, dtype=other_dtype, device=device)
                    if isinstance(v, tuple) else v for k, v in op_kwargs.items()},
                    sample_name=op_info.name + f'_{dtype}_promotion_{other_dtype}',
                )

def basic_sample_inputs_mint_optim_sgd(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    train_batchs = 10
    in_features = 3
    out_features = 4
    test_batchs = 20
    params={
        'lr':[True, False, 0.01], "momentum":[True, False, 0.07], "weight_decay":[0, 0.01],
         "dampening": [True, 0]
    }
    for lr, momentum, weight_decay, dampening in itertools.product(
        params["lr"], params["momentum"], params["weight_decay"], params["dampening"]
    ):
        yield OpSampleInput(
            op_input=make_arg(shape=(test_batchs, in_features)),
            op_args=(make_arg(shape=(out_features, in_features)),
                     make_arg(shape=(out_features,)),
                     make_arg(shape=(train_batchs, in_features)),
                     make_arg(shape=(train_batchs, out_features))),
            op_kwargs={"lr": lr, "momentum":momentum, "weight_decay":weight_decay, "dampening":dampening,
                       "nesterov":False, "maximize":True},
            sample_name=f"{op_info.name}_{dtype}"
        )
    yield OpSampleInput(
        op_input=make_arg(shape=(test_batchs, in_features)),
        op_args=(make_arg(shape=(out_features, in_features)),
                 make_arg(shape=(out_features,)),
                 make_arg(shape=(train_batchs, in_features)),
                 make_arg(shape=(train_batchs, out_features))),
        op_kwargs={"lr": 0.01, "momentum": 0.07, "weight_decay": 1.0, "dampening": 0,
                   "nesterov": True, "maximize": False},
        sample_name=f"{op_info.name}_{dtype}"
    )

def extra_sample_inputs_mint_optim_sgd(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    #empty shape
    yield OpSampleInput(
        op_input=make_arg(shape=(0, 3)),
        op_args=(make_arg(shape=(4, 3)),
                 make_arg(shape=(4,)), make_arg(shape=(0, 3)), make_arg(shape=(0, 4))),
        op_kwargs={"lr": 0.01},
        sample_name=f"{op_info.name}_{dtype}_empty_shape"
    )
    #large shape
    yield OpSampleInput(
        op_input=make_arg(shape=(1024, 128)),
        op_args=(make_arg(shape=(256, 128)),
                 make_arg(shape=(256,)), make_arg(shape=(1024, 128)), make_arg(shape=(1024,256))),
        op_kwargs={"lr": 0.01},
        sample_name=f"{op_info.name}_{dtype}_large_shape"
    )
    #small/large values
    values = small_large_values(dtype)
    yield OpSampleInput(
        op_input=ms.Tensor(np.random.choice(values,size=(10, 3)), dtype=dtype),
        op_args=(make_arg(shape=(4, 3)),
                 make_arg(shape=(4,)),
                 make_arg(shape=(10, 3)), make_arg(shape=(10, 4))),
        op_kwargs={"lr": 0.01},
        sample_name=f"{op_info.name}_{dtype}_small_large_values"
    )
    #extremal values
    values=[np.inf, -np.inf, np.nan, 1024]
    if dtype in [ms.float32, ms.float64]:
        yield OpSampleInput(
            op_input=ms.Tensor(np.random.choice(values, size=(10, 3)), dtype=dtype),
            op_args=(make_arg(shape=(4, 3)),
                     make_arg(shape=(4,)),
                     make_arg(shape=(10, 3)), make_arg(shape=(10, 4))),
            op_kwargs={"lr": 0.01},
            sample_name=f"{op_info.name}_{dtype}_extremal_values"
        )
    #discontiguous_values
    yield OpSampleInput(
        op_input=make_arg(shape=(10, 3), discontiguous=True),
        op_args=(make_arg(shape=(4, 3), discontiguous=True),
                 make_arg(shape=(4,), discontiguous=True),
                 make_arg(shape=(10, 3), discontiguous=True), make_arg(shape=(10, 4), discontiguous=True)),
        op_kwargs={"lr": 0.01},
        sample_name=f"{op_info.name}_{dtype}_discontiguous_True"
    )

def error_sample_inputs_mint_optim_sgd(op_info:OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(10, 3)),
            op_args=(make_arg(shape=(4, 3)),
                     make_arg(shape=(4,)),
                     make_arg(shape=(10, 3)), make_arg(shape=(10, 4))),
            op_kwargs={"lr": -0.05},
            sample_name=op_info.name
        ),
        op_error_type=ValueError,
        op_error_info="lr is less than 0",
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(10, 3)),
            op_args=(make_arg(shape=(4, 3)),
                     make_arg(shape=(4,)),
                     make_arg(shape=(10, 3)), make_arg(shape=(10, 4))),
            op_kwargs={"lr": 0.05, "momentum": -0.05, "nesterov":True},
            sample_name=op_info.name
        ),
        op_error_type=ValueError,
        op_error_info="momentum is less than 0 while nesterov is True",
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(10, 3)),
            op_args=(make_arg(shape=(4, 3)),
                     make_arg(shape=(4,)),
                     make_arg(shape=(10, 3)), make_arg(shape=(10, 4))),
            op_kwargs={"lr": 0.05, "momentum": 0.05, "nesterov": True, "dampening": 0.01},
            sample_name=op_info.name
        ),
        op_error_type=ValueError,
        op_error_info="dampening is not 0 while nesterov is True",
    )
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg(shape=(10, 3)),
            op_args=(make_arg(shape=(4, 3)),
                     make_arg(shape=(4,)),
                     make_arg(shape=(10, 3)), make_arg(shape=(10, 4))),
            op_kwargs={"lr": 0.05, "weight_decay":-0.05},
            sample_name=op_info.name
        ),
        op_error_type=ValueError,
        op_error_info="weight_decay is less than 0",
    )

def extra_sample_inputs_optim_adam(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    weight = make_arg((10, 3, 2, 2))
    net = Net(3)
    net.conv.weight.data = weight
    net.optimizer = mint.optim.Adam(net.trainable_params(),)

    yield from make_extra_tensor(op_info, (2, 3, 4, 5), (net, ),
                                 {}, dtype=dtype, device=device)


def extra_sample_inputs_optim_adamw(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    weight = make_arg((10, 3, 2, 2))

    yield from make_extra_tensor(op_info, (2, 3, 4, 5), (weight, ),
                                 {}, dtype=dtype, device=device, exclude_value=(-13141.5, 13141.5))


def basic_op_sample_inputs_grad_position_cover(basic_func, no_grad_dtype):
    def basic_op_sample_inputs(op_info,dtype, device=None, **input_kwargs):
        for sample in basic_func(op_info, dtype, device, **input_kwargs):
            if dtype in no_grad_dtype:
                sample.grad_position=()
            yield sample
    return basic_op_sample_inputs

def basic_sample_inputs_optim_adam(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        (SMALL_DIM_SIZE, MEDIUM_DIM_SIZE, LARGE_DIM_SIZE),
        (SMALL_DIM_SIZE, MEDIUM_DIM_SIZE, LARGE_DIM_SIZE, SMALL_DIM_SIZE),
    )


    for input_shape in cases:

        for lr, betas, eps, weight_decay, amsgrad, maximize in itertools.product([0.003, 0.004], [(0.9, 0.999)],
                                            [1e-8,], [0.0,], [False, True], [False, True]):
            weight =  make_arg((10, input_shape[-3], 2, 2),)
            net = Net(input_shape[-3])
            net.conv.weight.data = weight
            net.optimizer = mint.optim.Adam(net.trainable_params(), lr=lr, betas=betas, eps=eps,
                                            weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
            yield OpSampleInput(
            op_input=make_arg(input_shape),
            op_args=(net, ),
            op_kwargs={'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay, 'amsgrad': amsgrad,
                       'maximize': maximize},
            sample_name=op_info.name,
        )

def basic_sample_inputs_optim_adamw(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        (MEDIUM_DIM_SIZE, SMALL_DIM_SIZE, LARGE_DIM_SIZE, MEDIUM_DIM_SIZE),
        (SMALL_DIM_SIZE, MEDIUM_DIM_SIZE, LARGE_DIM_SIZE, SMALL_DIM_SIZE),
    )

    for input_shape in cases:

        for lr, betas, eps, weight_decay, amsgrad, maximize in itertools.product([0.003, 0.004], [(0.9, 0.999)],
                                            [1e-8,], [0.0,], [False, True], [False, True]):

            yield OpSampleInput(
                op_input=make_arg(input_shape),
                op_args=(make_arg((10,input_shape[-3], 2, 2)),),
                op_kwargs={'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay, 'amsgrad': amsgrad,
                           'maximize': maximize},
                sample_name=op_info.name,
            )

def error_sample_inputs_optim_adam(op_info: OpInfo, dtype=None, device=None, **kwargs):


    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    weight = make_arg((10, 3, 2, 2))
    net = Net(3)
    net.conv.weight.data = weight

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(net, ),
            op_kwargs={'lr': -0.5},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='lr should >=0',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(net, ),
            op_kwargs={'eps': -0.5},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='eps should >=0',
    )

    betas_errors=[(-0.01, 0.07), (0.07, -0.01), (1.0, 0.05), (0.05, 1.0)]

    for betas_error in betas_errors:

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((2, 3, 4, 5)),
                op_args=(net, ),
                op_kwargs={'betas': betas_error},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info='betas should >=0 and <1',
        )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(net, ),
            op_kwargs={'weight_decay': -0.5},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='weight_decay should >=0',
    )

def error_sample_inputs_optim_adamw(op_info: OpInfo, dtype=None, device=None, **kwargs):

    make_arg = functools.partial(make_tensor, device=device, dtype=ms.float32)

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(make_arg((10, 3, 2, 2)), ),
            op_kwargs={'lr': -0.5},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='lr should >=0',
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(make_arg((10, 3, 2, 2)), ),
            op_kwargs={'eps': -0.5},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='eps should >=0',
    )

    betas_errors=[(-0.01, 0.07), (0.07, -0.01), (1.0, 0.05), (0.05, 1.0)]

    for betas_error in betas_errors:

        yield OpErrorInput(
            op_sample_input=OpSampleInput(
                op_input=make_arg((2, 3, 4, 5)),
                op_args=(make_arg((10, 3, 2, 2)), ),
                op_kwargs={'betas': betas_error},
                sample_name=op_info.name,
            ),
            op_error_type=ValueError,
            op_error_info='betas should >=0 and <1',
        )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_arg((2, 3, 4, 5)),
            op_args=(make_arg((10, 3, 2, 2)), ),
            op_kwargs={'weight_decay': -0.5},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='weight_decay should >=0',
    )

def basic_unary_op_func_tensor_to_1(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    shapes = (
        (),
        (S,),
        (M, S),
        (S, S, L),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )
    for non_blocking in [True, False]:
        for copy in [True, False]:
            for input_shape in shapes:
                _input = make_func(input_shape, low=op_info.input_low,
                                   high=op_info.input_high, exclude_value=op_info.input_exclude_value)
                slen = len(input_shape)
                yield OpSampleInput(
                    _input,
                    op_args=(),
                    op_kwargs={'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f"{op_info.name}_{dtype}_{slen}d",
                )

def extra_unary_op_func_tensor_to_1(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    #discontiguous/empty tensors/small medium large size
    shapes = (
        (1,),
        (3, 1),
        (1, 2, 3),
        (0,),  # empty tensors
        (1, 0, 4),  # empty tensors
        (20,),  # 1D tensors with small size
        (812,),  # 1D tensors with medium size
        (1029, 918),  # 2D tensors with large size
    )
    for non_blocking in [True, False]:
        for copy in [True, False]:
            for input_shape in shapes:
                _input = make_func(input_shape, low=op_info.input_low,
                                   high=op_info.input_high, exclude_value=op_info.input_exclude_value)
                yield OpSampleInput(
                    _input,
                    op_args=(),
                    op_kwargs={'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f"{op_info.name}_shape_{input_shape}",
                )

    # +-inf nan
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    extremal_cases =([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])
    for non_blocking in [True, False]:
        for copy in [True, False]:
            for extremal_val, name_suffix in extremal_cases:
                t = make_func((5,))
                t[2] = extremal_val
                yield OpSampleInput(
                    op_input=t,
                    op_args=(),
                    op_kwargs={'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
                )


def basic_unary_op_func_tensor_to_2(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    shapes = (
        (),
        (S,),
        (M, S),
        (S, S, L),
        (2, 1, 3, 2),
        (2, 3, 4, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2, 1),
        (2, 1, 2, 2, 1, 2, 1, 2),
    )

    for non_blocking in [True, False]:
        for copy in [True, False]:
            for input_shape in shapes:
                _input = make_func(input_shape, low=op_info.input_low,
                                   high=op_info.input_high, exclude_value=op_info.input_exclude_value)
                slen = len(input_shape)
                yield OpSampleInput(
                    _input,
                    op_args=(),
                    op_kwargs={'device':'cpu', 'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f"{op_info.name}_{dtype}_{slen}d",
                )

def extra_unary_op_func_tensor_to_2(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype,)

    #discontiguous/empty tensors/small medium large size
    shapes = (
        (1,),
        (3, 1),
        (1, 2, 3),
        (0,),  # empty tensors
        (1, 0, 4),  # empty tensors
        (20,),  # 1D tensors with small size
        (812,),  # 1D tensors with medium size
        (1029, 918),  # 2D tensors with large size
    )

    for non_blocking in [True, False]:
        for copy in [True, False]:
            for input_shape in shapes:
                _input = make_func(input_shape, low=op_info.input_low,
                                   high=op_info.input_high, exclude_value=op_info.input_exclude_value)
                yield OpSampleInput(
                    _input,
                    op_args=(),
                    op_kwargs={'device':'cpu', 'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f"{op_info.name}_shape_{input_shape}",
                )

    # +-inf nan
    if device == 'ascend':
        ascend_name = MSContext.get_instance().get_ascend_soc_version()
        if ascend_name == 'ascend910':
            warnings.warn("Inf and NaN are unsupported on current Ascend devices.")
            return

    extremal_cases =([
        (float('nan'), 'nan'),
        (float('inf'), 'inf'),
        (float('-inf'), 'neg_inf'),
    ])
    for non_blocking in [True, False]:
        for copy in [True, False]:
            for extremal_val, name_suffix in extremal_cases:
                t = make_func((5,))
                t[2] = extremal_val
                yield OpSampleInput(
                    op_input=t,
                    op_args=(),
                    op_kwargs={'device':'cpu', 'dtype': dtype, 'non_blocking': non_blocking, 'copy': copy},
                    sample_name=f'{op_info.name}_{dtype}_default_extremal_{name_suffix}'
                )

def _generate_tensor_is_contiguous_contiguous_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate contiguous tensor inputs for tensor.is_contiguous.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering contiguous/non-contiguous memory layouts.
    """

    shapes = (
        (1,),
        (3, 1),
        (1, 2, 3),
    )

    for shape, discontiguous in itertools.product(shapes, [False]):
        yield OpSampleInput(
            op_input=make_tensor(shape, dtype=dtype, device=device, discontiguous=discontiguous),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_discontiguous_tensor_inputs",
        )

def extra_sample_inputs_tensor_is_contiguous_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate comprehensive reference inputs for tensor.is_contiguous.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Aggregated samples from multiple input generators.
    """
    if dtype in dtypes_extra_uint:
        return
    # tensors with many kinds of shapes
    yield from _generate_unary_op_tensors_sample_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with small value
    if dtype != ms.bool_:
        yield from _generate_unary_op_small_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with large value
    if dtype not in (ms.bool_, ms.uint8, ms.int8):
        yield from _generate_unary_op_large_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # contiguous tensors
    yield from _generate_tensor_is_contiguous_contiguous_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensor with extremal value
    if dtype.is_floating_point or dtype.is_complex:
        yield from _generate_unary_op_extremal_value_tensor_inputs_func(op_info, dtype, device, **kwargs)


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

def basic_reference_inputs_tensor_from_numpy(op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    for dtypes in [np.float32, np.float64, np.complex64, np.complex128, np.float16, np.int64, np.int32, np.int16,
                   np.int8, np.uint8, np.bool_]:
        yield OpSampleInput(
            op_input=np.array([1, 2], dtype=dtypes),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_1d",
        )
        yield OpSampleInput(
            op_input=np.array([[1, 2],[3, 4]], dtype=dtypes),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_2d",
        )
        yield OpSampleInput(
            op_input=np.array([[[1, 2], [3, 4]]], dtype=dtypes),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_3d",
        )
        yield OpSampleInput(
            op_input=np.random.rand(2, 3, 4, 5).astype(dtypes),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_4d",
        )
        yield OpSampleInput(
            op_input=np.random.rand(1, 2, 3, 4, 5).astype(dtypes),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_5d",
        )
        yield OpSampleInput(
            op_input=np.random.rand(1, 2, 3, 4, 5, 6).astype(dtypes),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_6d",
        )
        yield OpSampleInput(
            op_input=np.array(42),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_scalar",
        )
        yield OpSampleInput(
            op_input=np.array([]),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_empty",
        )

def extra_reference_inputs_tensor_from_numpy(op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    #small/large values
    values = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp32_large_values']
    for value in values:
        yield OpSampleInput(
            op_input=np.array([value], dtype=np.float32),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_small_large_value",
        )
    #extremal values
    for value in [np.inf, np.nan,-np.inf]:
        yield OpSampleInput(
            op_input=np.array([value], dtype=np.float32),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_value",
        )
    yield OpSampleInput(
        op_input=np.random.rand(1024, 1024).astype(np.float32),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_large_value",
    )
    yield OpSampleInput(
        op_input=np.array([[-1, -2], [-3, -4]], dtype=np.float32),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_elements_negative",
    )


def error_reference_inputs_tensor_from_numpy(op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=[1, 1, 1],
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_input_list",
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info=f"{op_info.name}_input_list",
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=None,
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_none",
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info=f"{op_info.name}_none",
    )

    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input="array",
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_not_support_dtype",
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info=f"{op_info.name}_not_support_dtype",
    )

def basic_reference_inputs_tensor_item(op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    yield OpSampleInput(
        op_input=make_tensor((1,), dtype=dtype, device=device),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_1d",
    )
    yield OpSampleInput(
        op_input=make_tensor((), dtype=dtype, device=device),
        op_args=(),
        op_kwargs={},
        sample_name=f"{op_info.name}_{dtype}_scalar",
    )

def extra_reference_inputs_tensor_item(op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    #small/large values
    values = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp32_large_values']
    for value in values:
        yield OpSampleInput(
            op_input=ms.Tensor([value], dtype=dtype),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_small_large_value",
        )
    #extremal values
    for value in [np.inf, np.nan,-np.inf]:
        yield OpSampleInput(
            op_input=ms.Tensor([value], dtype=dtype),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_{dtype}_extremal_value",
        )

def error_reference_inputs_tensor_item(op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    #1d with more than 1 elements
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=ms.Tensor([1, 2], dtype = ms.float32),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_double_eles",
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info=f"{op_info.name}_double_eles",
    )
    # 1d with more than 1-d
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=ms.Tensor([[1], [2]], dtype=ms.float32),
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_ambiguous_2d",
        ),
        op_error_type=(ValueError, TypeError),
        op_error_info=f"{op_info.name}_ambiguous_2d",
    )

def error_reference_inputs_tensor_nbytes(op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    # tensor list
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=[ms.Tensor([1, 2], dtype = ms.float32)],
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_list_tensor",
        ),
        op_error_type=(ValueError, TypeError, AttributeError),
        op_error_info=f"{op_info.name}_list_tensor",
    )
    # not tensor
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=1.0,
            op_args=(),
            op_kwargs={},
            sample_name=f"{op_info.name}_not_tensor",
        ),
        op_error_type=(ValueError, TypeError, AttributeError),
        op_error_info=f"{op_info.name}_not_tensor",
    )
