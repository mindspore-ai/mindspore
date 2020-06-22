# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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

"""Basic composite operations."""
from functools import partial

from mindspore import context
from ..._c_expression import EnvInstance_, GradOperation_, HyperMap_, Map_, MultitypeFuncGraph_, Tail_, \
                             TupleAdd_, TupleSlice_, UnpackCall_, ZipOperation_, ListAppend_, TupleGetItemTensor_
from ...common import dtype as mstype
from ...common.api import ms_function, _pynative_exec, _wrap_func
from .. import functional as F
from ...common.parameter import Parameter


__all__ = [EnvInstance_, TupleAdd_, TupleSlice_, UnpackCall_, TupleGetItemTensor_]


def add_flags(fn=None, **flags):
    """
    An decorator to add flag for a function.

    Note:
        Only supports bool value.

    Args:
        fn (Function): Function or cell to add flag. Default: None.
        flags (dict): Flags use kwargs. Default: None.

    Returns:
        Function, the fn added flags.

    Examples:
        >>> add_flags(net, predit=True)
    """
    def deco(fn):
        # need set the attr and access on c++
        if not hasattr(fn, "_mindspore_flags"):
            fn._mindspore_flags = {}

        fn._mindspore_flags.update({**flags})
        return fn
    ret = deco
    if fn is not None:
        ret = deco(fn)
    return ret


def core(fn=None, **flags):
    """
    A decorator to add flag to a function.

    By default, the function is marked core=True using this decorator to
    set flag to a graph.

    Args:
        fn (Function): Function to add flag. Default: None.
        flags (dict): The following flags can be set core, which indicates that this is a core function or
                      other flag. Default: None.
    """
    # need set the attr and access on c++

    def deco(fn):
        fn._mindspore_flags = {
            'core': True,
            **flags,
        }
        return fn

    if fn is not None:
        ret = deco(fn)
    else:
        ret = deco
    return ret


class GradOperation(GradOperation_):
    """
    An metafuncgraph object which is used to get the gradient of output of a network(function).

    The GradOperation will convert the network(function) into a back propagation graph.

    Args:
        get_all (bool): If True, get all the gradients w.r.t inputs. Default: False.
        get_by_list (bool): If True, get all the gradients w.r.t Parameter variables.
            If get_all and get_by_list are both False, get the gradient w.r.t first input.
            If get_all and get_by_list are both True, get the gradients w.r.t inputs and Parameter variables
            at the same time in the form of ((grads w.r.t inputs), (grads w.r.t parameters)). Default: False.
        sens_param (bool): Whether append sensitivity as input. If sens_param is False,
            a 'ones_like(outputs)' sensitivity will be attached automatically. Default: False.
    """

    def __init__(self, name,
                 get_all=False, get_by_list=False, sens_param=False):
        self.get_all = get_all
        self.get_by_list = get_by_list
        self.sens_param = sens_param
        GradOperation_.__init__(self, name, get_all, get_by_list, sens_param)
        self.grad_fn = None
        self.fn = None
        self.need_forward = False

    def __call__(self, fn, weights=None):
        grad_ = GradOperation('grad', self.get_all, self.get_by_list, self.sens_param)
        if self.grad_fn is None or self.fn != fn:
            if self.get_by_list:
                if context.get_context("mode") == context.GRAPH_MODE:
                    @ms_function(obj=fn)
                    def after_grad(*args):
                        return grad_(fn, weights)(*args)
                else:
                    @_wrap_func
                    def after_grad(*args):
                        if fn.is_run and not fn.requires_grad:
                            raise ValueError("obj must set_grad.")
                        if not fn.is_run:
                            self.need_forward = True
                            print("already has forward run before grad by user")
                        if self.need_forward:
                            fn.set_grad()
                            if self.sens_param:
                                f_args = args[:-1]
                                fn(*f_args)
                            else:
                                fn(*args)
                        _pynative_exec.grad(grad_, fn, weights, *args)
                        out = _pynative_exec(*args)
                        _pynative_exec.clear()
                        return out
            else:
                @ms_function(obj=fn)
                def after_grad(*args):
                    return grad_(fn)(*args)
            self.grad_fn = after_grad
            self.fn = fn
        return self.grad_fn


grad = GradOperation('grad')
grad_all = GradOperation('get_all', get_all=True)
grad_by_list = GradOperation('get_by_list', get_by_list=True)
grad_with_sens = GradOperation('grad_with_sens', sens_param=True)
grad_all_with_sens = GradOperation('grad_all_with_sens', get_all=True, sens_param=True)
grad_by_list_with_sens = GradOperation('grad_by_list_with_sens', get_by_list=True, sens_param=True)


class MultitypeFuncGraph(MultitypeFuncGraph_):
    """
    Generate multiply graph.

    MultitypeFuncGraph is a class used to generate graphs for function with different type as input.

    Args:
        name (str): Operator name.

    Raises:
        ValueError: Cannot find matching fn for the given args.

    Examples:
        >>> # `add` is a metagraph object which will add two objects according to
        >>> # input type using ".register" decorator.
        >>> add = MultitypeFuncGraph('add')
    """

    def __init__(self, name):
        MultitypeFuncGraph_.__init__(self, name)
        self.entries = list()

    def __call__(self, *args):
        def unwrap(arg):
            if isinstance(arg, Parameter):
                return arg.data
            return arg
        types = tuple(map(lambda arg: mstype.get_py_obj_dtype(unwrap(arg)), args))
        for sigs, fn in self.entries:
            if len(sigs) != len(types):
                continue
            if any(not mstype.issubclass_(type_, sig) for sig, type_ in zip(sigs, types)):
                continue
            output = fn(*args)
            return output
        raise ValueError("Cannot find fn match given args.")

    def register(self, *type_names):
        """Register a function for the given type string."""
        def deco(fn):
            types = tuple(map(mstype.typing.str_to_type, type_names))
            self.register_fn(type_names, fn)
            self.entries.append((types, fn))
            return fn
        return deco


class HyperMap(HyperMap_):
    """
    Hypermap will apply the set operation on input sequences.

    Which will apply the operations of every elements of the sequence.

    Args:
        ops (Union[MultitypeFuncGraph, None]): `ops` is the operation to apply. If `ops` is `None`,
            the operations should be putted in the first input of the instance.

    Inputs:
        - **args** (Tuple[sequence]) - If `ops` is not `None`, all the inputs should be the same length sequences,
          and each row of the sequences. e.g. If args length is 2, and for `i` in length of each sequence
          `(args[0][i], args[1][i])` will be the input of the operation.

          If `ops` is not `None`, the first input is the operation, and the other is inputs.

    Outputs:
        sequence, the output will be same type and same length of sequence from input and the value of each element
        is the result of operation apply each row of element. e.g. `operation(args[0][i], args[1][i])`.
    """

    def __init__(self, ops=None):
        self.ops = ops
        if ops:
            HyperMap_.__init__(self, ops)
        else:
            HyperMap_.__init__(self)

    def __call__(self, *args):
        func = self.ops
        args_list = args
        hypermap = self
        if self.ops is None:
            func = args[0]
            args_list = args[1:]
            hypermap = partial(self, func)
        # is leaf
        if not isinstance(args_list[0], (tuple, list)):
            return func(*args_list)
        return tuple(map(hypermap, *args_list))


class Map(Map_):
    """
    Map will apply the set operation on input sequences.

    Which will apply the operations of every elements of the sequence.

    Args:
        ops (Union[MultitypeFuncGraph, None]): `ops` is the operation to apply. If `ops` is `None`,
            the operations should be putted in the first input of the instance.

    Inputs:
        - **args** (Tuple[sequence]) - If `ops` is not `None`, all the inputs should be the same length sequences,
          and each row of the sequences. e.g. If args length is 2, and for `i` in length of each sequence
          `(args[0][i], args[1][i])` will be the input of the operation.

          If `ops` is not `None`, the first input is the operation, and the other is inputs.

    Outputs:
        sequence, the output will be same type and same length of sequence from input and the value of each element
        is the result of operation apply each row of element. e.g. `operation(args[0][i], args[1][i])`.
    """

    def __init__(self, ops=None):
        self.ops = ops
        if ops:
            Map_.__init__(self, ops)
        else:
            Map_.__init__(self)

    def __call__(self, *args):
        func = self.ops
        args_list = args
        if self.ops is None:
            func = args[0]
            args_list = args[1:]
        return tuple(map(func, *args_list))


class _ListAppend(ListAppend_):
    """
    A metafuncgraph class that append one element to list.

    Args:
        name (str): The name of the metafuncgraph object.
    """
    def __init__(self, name):
        ListAppend_.__init__(self, name)

    def __call__(self, *args):
        pass


_append = _ListAppend("append")


class _Tail(Tail_):
    """
    A metafuncgraph class that generates tail elements of the tuple.

    Args:
        name (str): The name of the metafuncgraph object.
    """

    def __init__(self, name):
        Tail_.__init__(self, name)

    def __call__(self, *args):
        pass


tail = _Tail('tail')


class _ZipOperation(ZipOperation_):
    """Generates a tuple of zip iterations for inputs."""

    def __init__(self, name):
        ZipOperation_.__init__(self, name)

    def __call__(self, *args):
        pass


zip_operation = _ZipOperation('zip_operation')
"""`zip_operation` will generate a tuple of zip iterations of inputs."""


env_get = MultitypeFuncGraph("env_get")


@env_get.register("EnvType", "Tensor")
def _tensor_env_get(env, parameter):
    """Used to get env."""
    return F.env_getitem(env, F.ref_to_embed(parameter), F.zeros_like(parameter))
