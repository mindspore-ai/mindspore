# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Implementation of Numerical gradients checking."""
# pylint: disable=missing-docstring

from typing import Callable, List, Any

import numpy as np
import mindspore._c_expression as _c_expression

from mindspore import ParameterTuple
from mindspore import Tensor
from mindspore import context
from mindspore.ops.composite import GradOperation
from .block_util import get_output_cell, gen_net, gen_grad_net, \
    get_uniform_with_shape, set_block_phase, get_output_reduce_cell, set_block_param_with_rand


class _GradChecker:
    """
    Check the theoretical Jacobian against numeric

    Arguments:
        fn: The function under test.
        gfn: The high order function to compute the derivative function.
        args: The point in the function's domain where we want
            to estimate the gradient.

    """

    def __init__(self,
                 fn: Callable,
                 grad_wraper: GradOperation,
                 args: List[Any],
                 delta: float = 1e-3,
                 max_error: float = 1e-3,
                 input_selector=None,
                 output_selector=None,
                 sampling_times=-1,
                 reduce_output=False) -> None:
        """Initialize a GradChecker."""
        self.delta = delta
        self.scale = 2 * delta
        self.max_error = max_error
        self.sampling_times = sampling_times

        self.fn = self.prepare_func(fn)

        self.args = args
        out = self.fn(*self.args)
        self.out = self.wrap(out)

        self.nin = len(self.args)
        self.nout = len(self.out)
        self.gfns = []

        if reduce_output:
            fn = get_output_reduce_cell(fn, self.nout)
            self.fn = self.prepare_func(fn)
            out = self.fn(*self.args)
            self.out = self.wrap(out)

        if self.nout == 0:
            raise Exception(f'number of outputs expected to be >=1, but got {self.nout}')

        if self.nout == 1:
            self.gfns.append(self.prepare_func(fn, grad_wraper))
        else:
            for i in range(self.nout):
                cell = get_output_cell(fn, self.nin, i)
                self.gfns.append(self.prepare_func(cell, grad_wraper))

        self.input_selector = input_selector
        self.adjust_input_selector()
        if output_selector:
            self.output_selector = output_selector
        else:
            self.output_selector = [i for i in range(self.nout)]

    def adjust_input_selector(self):
        raise Exception('Not implemented')

    def sampling(self, superset):
        # -1 stands for all
        if self.sampling_times == -1 or self.sampling_times >= len(superset):
            return superset
        np.random.seed(0)
        ret = np.random.choice(superset, self.sampling_times, replace=False)
        return list(ret)

    def prepare_func(self, f, grad_wraper=None):
        """Return a function that executes 'f'.

        Args:
        f: the function.
        grad_wraper: grad op

        Returns:
        a function that will be evaluated in both Graph and PyNative mode
        """
        set_block_param_with_rand(f, get_uniform_with_shape)

        if context.get_context("mode") == context.PYNATIVE_MODE:
            if grad_wraper:
                def func_backward_pynative(*inputs):
                    net = gen_grad_net(f, grad_wraper, len(inputs) - 1, inputs[-1])

                    def _func_pynative(*inputs):
                        return net(*inputs)

                    return _func_pynative(*(inputs[:-1]))

                return func_backward_pynative

            def func_forward_pynative(*inputs):
                net = gen_net(f, len(inputs))

                def _func_pynative(*inputs):
                    return net(*inputs)

                return _func_pynative(*inputs)

            return func_forward_pynative

        if grad_wraper:
            def func_backward_graph(*inputs):
                set_block_phase(f, 'train')
                net = gen_grad_net(f, grad_wraper, len(inputs) - 1, inputs[-1])
                return net(*(inputs[:-1]))

            return func_backward_graph

        def func_forward_graph(*inputs):
            set_block_phase(f, 'predict')
            net = gen_net(f, len(inputs))
            return net(*inputs)

        return func_forward_graph

    def to_numpy(self, x):
        if isinstance(x, (Tensor, _c_expression.Tensor)):
            return x.asnumpy()
        return x

    def to_numpy_and_scale(self, x):
        if isinstance(x, (Tensor, _c_expression.Tensor)):
            return x.asnumpy() * self.delta
        return x * self.delta

    def wrap(self, x):
        if isinstance(x, tuple):
            return x
        return (x,)

    def get_sens(self, i):
        raise Exception('Not implemented')

    def get_ith_elem(self, c, i):
        if isinstance(c, (list, tuple)):
            return c[i]
        return c

    def compute_theoretical(self, i):
        args = list(self.args)
        args.append(self.get_sens(i))

        print('GradChecker.compute_theoretical.args', args)
        gout = self.gfns[i](*args)
        gout = self.wrap(gout)
        self.gout = [self.to_numpy_and_scale(g) if isinstance(g, _c_expression.Tensor) \
                         else self.to_numpy_and_scale(np.array(g)) for g in gout]
        print('GradChecker.compute_theoretical.gout', self.gout)

    def check_against_numeric(self, out_index):
        raise Exception('Not implemented')

    def check_against_numeric_one_step(self, args, index, out_index):
        if isinstance(args, ParameterTuple):
            x = args[index].data.asnumpy()
        else:
            x = args[index]
        x_shape = x.shape
        x_size = np.product(x_shape)
        for row in self.sampling(list(range(x_size))):
            original = x.ravel().view()[row]
            x.ravel().view()[row] += self.delta
            y_pos = self.to_numpy_and_scale(self.get_ith_elem(self.fn(*self.args), out_index))
            x.ravel().view()[row] = original
            x.ravel().view()[row] -= self.delta
            y_neg = self.to_numpy_and_scale(self.get_ith_elem(self.fn(*self.args), out_index))
            x.ravel().view()[row] = original
            diff = (y_pos - y_neg) / self.scale
            numeric_grad = diff.sum()
            insert_virtual_grad = False
            if numeric_grad == 0 and not insert_virtual_grad:
                self.gout.insert(0, 0)
                insert_virtual_grad = True
                continue
            theoretical_grad = self.gout[index].ravel().view()[row]

            if np.fabs(numeric_grad - theoretical_grad).max() > self.max_error:
                raise Exception(f'Gradients of df{out_index}/darg{index},{row} do not match, '
                                f'expect {numeric_grad}, actual {theoretical_grad}')

            print(f'GradChecker.check_against_numeric.numeric df{out_index}/darg{index}: '
                  f'{numeric_grad}, theoretical: {theoretical_grad}')

    # approximate accuracy, but efficient
    def assert_match(self):
        print(f'==========================={self.fn.__name__}==================================')
        print('GradChecker.delta', self.delta)
        print('GradChecker.max_error', self.max_error)
        print('GradChecker.args', self.args)
        print('GradChecker.out', self.out)
        print('GradChecker.nin', self.nin)
        print('GradChecker.nout', self.nout)
        for i in self.output_selector:
            self.compute_theoretical(i)
            self.check_against_numeric(i)

    def check_against_numeric_jacobian(self, out_index):
        raise Exception('Not implemented')

    def check_against_numeric_jacobian_one_step(self, args, index, out_index):
        if isinstance(args, ParameterTuple):
            x = args[index].data.asnumpy()
        else:
            x = args[index]
        x_shape = x.shape
        x_size = np.product(x_shape)
        dy = self.to_numpy(self.get_sens(out_index))
        dy_size = np.product(dy.shape)
        numeric_jacobian = np.zeros((x_size, dy_size), dtype=self.to_numpy(x).dtype)
        for row in range(x_size):
            original = x.ravel().view()[row]
            x.ravel().view()[row] += self.delta
            y_pos = self.to_numpy_and_scale(self.get_ith_elem(self.fn(*self.args), out_index))
            x.ravel().view()[row] = original
            x.ravel().view()[row] -= self.delta
            y_neg = self.to_numpy_and_scale(self.get_ith_elem(self.fn(*self.args), out_index))
            x.ravel().view()[row] = original
            diff = (y_pos - y_neg) / self.scale
            numeric_jacobian[row, :] = diff.ravel().view(numeric_jacobian.dtype)

        dy_mask = np.zeros(dy.shape, dtype=dy.dtype)
        theoretical_jacobian = np.zeros((x_size, dy_size), dtype=self.to_numpy(x).dtype)
        for col in range(dy_size):
            col_jacobian = self.compute_theoretical_jacobian(index, out_index, dy_mask, col)
            theoretical_jacobian[:, col] = col_jacobian.ravel().view(theoretical_jacobian.dtype)

        if np.fabs(numeric_jacobian - theoretical_jacobian).max() > self.max_error:
            raise Exception(f'GradChecker.check_against_numeric_jacobian_one_step expect {out_index}/darg{index}: '
                            f'{numeric_jacobian}, actual: {theoretical_jacobian}')

        print(f'GradChecker.check_against_numeric_jacobian_one_step.numeric jacobian of output{out_index}/darg{index}: '
              f'{numeric_jacobian}, theoretical: {theoretical_jacobian}')

    def compute_theoretical_jacobian(self, index, out_index, dy_mask, jacobian_col):
        if (out_index, jacobian_col, index) in self.theoretical_jacobian_cache:
            return self.theoretical_jacobian_cache[(out_index, jacobian_col, index)]

        dy_mask.ravel().view()[jacobian_col] = 1.0
        args = list(self.args)
        args.append(Tensor(dy_mask))
        print('GradChecker.compute_theoretical.args', args)
        gout = self.wrap(self.gfns[out_index](*args))
        gout = [self.to_numpy_and_scale(g) if isinstance(g, _c_expression.Tensor) \
                    else self.to_numpy_and_scale(np.array(g)) for g in gout]
        print('GradChecker.compute_theoretical.gout', gout)
        dy_mask.ravel().view()[jacobian_col] = 0.0

        for i, g in enumerate(gout):
            self.theoretical_jacobian_cache[(out_index, jacobian_col, i)] = g

        return gout[index]

    # more accurate, but inefficient
    def assert_match_jacobian(self):
        print(f'==========================={self.fn.__name__}==================================')
        print('GradChecker.delta', self.delta)
        print('GradChecker.max_error', self.max_error)
        print('GradChecker.args', self.args)
        print('GradChecker.out', self.out)
        print('GradChecker.nin', self.nin)
        print('GradChecker.nout', self.nout)

        self.theoretical_jacobian_cache = {}
        for i in self.output_selector:
            self.check_against_numeric_jacobian(i)


class ScalarGradChecker(_GradChecker):
    def __init__(self,
                 fn: Callable,
                 args: List[Any],
                 delta: float = 1e-3,
                 max_error: float = 1e-3,
                 input_selector=None,
                 output_selector=None,
                 sampling_times=-1,
                 reduce_output=False) -> None:
        grad_op = GradOperation(get_all=True, sens_param=True)
        super(ScalarGradChecker, self).__init__(fn, grad_op, args, delta, max_error, input_selector, \
                                                output_selector, sampling_times, reduce_output)

    def adjust_input_selector(self):
        if not self.input_selector:
            self.input_selector = [i for i in range(self.nin)]

    def get_sens(self, i):
        return 1.0

    def check_against_numeric(self, out_index):
        args = list(self.args)
        for i in self.sampling(self.input_selector):
            print(f'GradChecker.check_against_numeric.args[{i}]', args[i])
            args_pos = args[:i] + [args[i] + self.delta] + args[i + 1:]
            args_neg = args[:i] + [args[i] - self.delta] + args[i + 1:]
            y_pos = self.to_numpy_and_scale(self.get_ith_elem(self.fn(*args_pos), out_index))
            y_neg = self.to_numpy_and_scale(self.get_ith_elem(self.fn(*args_neg), out_index))
            diff = (y_pos - y_neg) / self.scale

            if np.fabs(diff - self.gout[i]).max() > self.max_error:
                raise Exception(f'Gradients of df{out_index}/darg{i} do not match,'
                                f'expect {diff}, actual {self.gout[i]}')

            print(f'GradChecker.check_against_numeric.numeric df{out_index}/darg{i}: {diff}, '
                  f'theoretical: {self.gout[i]}')

    # for scalar, jacobian is same with gradient
    def assert_match_jacobian(self):
        self.assert_match()


class OperationGradChecker(_GradChecker):
    def __init__(self,
                 fn: Callable,
                 args: List[Any],
                 delta: float = 1e-3,
                 max_error: float = 1e-3,
                 input_selector=None,
                 output_selector=None,
                 sampling_times=-1,
                 reduce_output=False) -> None:
        grad_op = GradOperation(get_all=True, sens_param=True)
        super(OperationGradChecker, self).__init__(fn, grad_op, args, delta, max_error, input_selector, \
                                                   output_selector, sampling_times, reduce_output)

    def get_sens(self, i):
        return Tensor(np.ones_like(self.out[i].asnumpy()))

    def adjust_input_selector(self):
        if not self.input_selector:
            self.input_selector = [i for i in range(self.nin)]

    def check_against_numeric(self, out_index):
        args = [self.to_numpy(arg) for arg in self.args]
        for i in self.input_selector:
            self.check_against_numeric_one_step(args, i, out_index)

    def check_against_numeric_jacobian(self, out_index):
        args = [self.to_numpy(arg) for arg in self.args]
        for i in self.input_selector:
            self.check_against_numeric_jacobian_one_step(args, i, out_index)


class NNGradChecker(_GradChecker):
    def __init__(self,
                 fn: Callable,
                 args: List[Any],
                 delta: float = 1e-3,
                 max_error: float = 1e-3,
                 input_selector=None,
                 output_selector=None,
                 sampling_times=-1,
                 reduce_output=False) -> None:
        grad_op = GradOperation(get_by_list=True, sens_param=True)
        self.params = ParameterTuple(fn.trainable_params())
        super(NNGradChecker, self).__init__(fn, grad_op, args, delta, max_error, input_selector, \
                                            output_selector, sampling_times, reduce_output)

    def get_sens(self, i):
        return Tensor(np.ones_like(self.out[i].asnumpy()))

    def adjust_input_selector(self):
        if not self.input_selector:
            self.input_selector = [i for i in range(len(self.params))]

    def check_against_numeric(self, out_index):
        for i in self.input_selector:
            self.check_against_numeric_one_step(self.params, i, out_index)

    def check_against_numeric_jacobian(self, out_index):
        for i in self.input_selector:
            self.check_against_numeric_jacobian_one_step(self.params, i, out_index)


def check_gradient(fn, *args, delta=1e-3, max_error=1e-3,
                   grad_checker_class=OperationGradChecker,
                   input_selector=None,
                   output_selector=None,
                   sampling_times=-1,
                   reduce_output=False):
    """Check the theoretical Jacobian against numeric of `fn`.
    Args:
        fn: the function that might be scalar function, operation, or neural network.
        args: a list arguments for the function
        delta: (optional) perturbation used to compute numeric Jacobian.
        max_error: (optional) max_error that is allowed between theoretical and numeric.
        grad_checker_class: (optional) checker, default OperationGradChecker.
        input_selector: list of input index that will be checked against numeric
        output_selector: list of output index that will be checked against numeric
    """
    grad_checker = grad_checker_class(fn=fn,
                                      args=list(args),
                                      delta=delta,
                                      max_error=max_error,
                                      input_selector=input_selector,
                                      output_selector=output_selector,
                                      sampling_times=sampling_times,
                                      reduce_output=reduce_output)
    grad_checker.assert_match()


def check_jacobian(fn, *args, delta=1e-3, max_error=1e-3,
                   grad_checker_class=OperationGradChecker,
                   input_selector=None,
                   output_selector=None):
    """Check the theoretical Jacobian against numeric of `fn`.
    Args:
        fn: the function that might be scalar function, operation, or neural network.
        args: a list arguments for the function
        delta: (optional) perturbation used to compute numeric Jacobian.
        max_error: (optional) max_error that is allowed between theoretical and numeric.
        grad_checker_class: (optional) checker, default OperationGradChecker
        input_selector: list of input index that will be checked against numeric
        output_selector: list of output index that will be checked against numeric
    """
    grad_checker = grad_checker_class(fn=fn,
                                      args=list(args),
                                      delta=delta,
                                      max_error=max_error,
                                      input_selector=input_selector,
                                      output_selector=output_selector)
    grad_checker.assert_match_jacobian()
