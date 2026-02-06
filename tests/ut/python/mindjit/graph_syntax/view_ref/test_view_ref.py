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

"""tesing view's loss ref by resize"""

import copy
import os
import pytest

import numpy as np
import mindspore as ms
from mindspore import Tensor, context, nn, ops, Parameter, mint


IR_LEVEL = 2
INT = 0
FLOAT = 1
BOOL = 2
TUPLE = 3
LIST = 4


class HelpNet(nn.Cell):
    def __init__(self, prim):
        super().__init__()
        self.op = prim

    # Inorder to run the net twice, the inputs with the type of list/tuple/scalar in replaced by help tensor
    def construct(self, *args):
        # the last two args indicates the index and type(tuple/list/scalar) of inputs which are replaced by help tensor
        index = args[-2]
        types = args[-1]

        new_args = list(args[:-2])
        for i, idx in enumerate(index):
            if types[i] == INT:
                new_args[idx] = int(args[idx])
            elif types[i] == FLOAT:
                new_args[idx] = float(args[idx])
            elif types[i] == BOOL:
                new_args[idx] = bool(args[idx])
            elif types[i] == TUPLE:
                new_args[idx] = ops.TensorToTuple()(args[idx])
            elif types[i] == LIST:
                new_args[idx] = ops.TensorToTuple()(args[idx])

        return self.op(*new_args)


def is_numerical_sequence(seq):
    if isinstance(seq, (tuple, list)):
        if seq:
            return isinstance(seq[0], (int, float))
        return True
    return False


def replace_nontensor_with_help_tensor(inputs):
    nontensor_input_index = []
    nontensor_input_type = []
    new_inputs = copy.deepcopy(inputs)
    for i, x in enumerate(inputs):
        if isinstance(x, tuple) and is_numerical_sequence(x):
            nontensor_input_type += [TUPLE]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x)
        elif isinstance(x, list) and is_numerical_sequence(x):
            nontensor_input_type += [LIST]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x)
        elif isinstance(x, int) and not isinstance(x, bool):
            nontensor_input_type += [INT]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x)
        elif isinstance(x, bool):
            nontensor_input_type += [BOOL]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x, dtype=ms.bool_)
        elif isinstance(x, float):
            nontensor_input_type += [FLOAT]
            nontensor_input_index += [i]
            new_inputs[i] = Tensor(x)
        elif x is not None and not isinstance(x, (Tensor, tuple, list, str)):
            raise TypeError(f"Unsupported type: {type(x)}")

    new_inputs += [tuple(nontensor_input_index), tuple(nontensor_input_type)]
    return new_inputs


def convert_tensor_to_dynamic(inputs, dynamic_type):
    new_inputs = copy.deepcopy(inputs)
    for i, x in enumerate(inputs):
        if isinstance(x, Tensor) and not isinstance(x, Parameter):
            ori_shape = x.shape
            if dynamic_type == 'DYNAMIC_SHAPE' and ori_shape:
                new_shape = [None for _ in ori_shape]
            else:
                new_shape = None

            new_input = Tensor(shape=new_shape, dtype=x.dtype)
            new_inputs[i] = new_input

        if isinstance(x, (tuple, list)):
            new_input = convert_tensor_to_dynamic(x, dynamic_type)
            new_inputs[i] = new_input

    return new_inputs


def run_with_dynamic_resize(prim, inputs_seq, ir_path, tensor_dynamic_type):
    context.set_context(save_graphs=IR_LEVEL, save_graphs_path=ir_path)
    compile_inputs = convert_tensor_to_dynamic(inputs_seq, 'DYNAMIC_RANK')
    compile_inputs = replace_nontensor_with_help_tensor(compile_inputs)
    run_inputs = replace_nontensor_with_help_tensor(inputs_seq)
    dynamic_net = HelpNet(prim)
    dynamic_net.set_inputs(*compile_inputs)
    dynamic_net(*clone_inputs(run_inputs))


def clone_inputs(args, inplace_update=False):
    def clone_func(arg):
        if isinstance(arg, (Tensor, Parameter)):
            return arg.copy()
        return copy.deepcopy(arg)

    if not inplace_update:
        return args
    return [clone_func(arg) for arg in args]


def TEST_VIEW(op, inputs_seq):
    context.set_context(mode=context.GRAPH_MODE, jit_level='O0')
    save_path = 'ir_view_ref'
    os.system(f"rm {save_path} -rf")
    ir_path = f"{save_path}/Resize"
    tensor_dynamic_type = ['DYNAMIC_SHAPE', 'DYNAMIC_RANK']
    run_with_dynamic_resize(op, inputs_seq, ir_path, tensor_dynamic_type)

    os.chdir(ir_path)
    for filename in os.listdir():
        if '_validate' in filename:
            check_nextline_followed_by_ref(filename, 'PrimFunc_NarrowView')

    os.system(f"rm -rf ../../{save_path}")


def check_nextline_followed_by_ref(filename, target):
    """
    look for the target string that appears for the first time
    and check that line below the string matches target
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            has_target = False
            for line in file:
                stripped_line = line.strip()
                if target in stripped_line:
                    next_line = next(file, None)
                    has_target = True
                    if next_line:
                        next_line = next_line.strip()
                        ref_count = next_line.count('Ref')
                        assert ref_count == 2, f"Expected 2 occurrences of 'Ref' in next_line, but found {ref_count}"
                        assert next_line is not None, "next_line should not be None"

            assert has_target, "target should be found in file"

    except FileNotFoundError:
        print(f"File {filename} not found.")


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def diff_forward_dyn_func(input_x, n=1, dim=-1, prepend=None, append=None):
    return mint.diff(input_x, n, dim, prepend, append)


@pytest.mark.skip(reason="Diff in graph mode not support implementation using the view op.")
def test_view_resize_loss_ref():
    """
    Feature: View
    Description: test view loss ref
    Expectation: expect correct result.
    """
    input_x = generate_random_input((4, 3, 5), np.float32)
    input_x1 = generate_random_input((4, 3, 6), np.float32)
    input_x2 = generate_random_input((4, 3, 7), np.float32)
    TEST_VIEW(diff_forward_dyn_func, [Tensor(input_x), 2, -1, Tensor(input_x1), Tensor(input_x2)])
