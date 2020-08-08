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

"""primitive"""

import inspect
import copy
from mindspore.common.api import _wrap_func
from mindspore.common import Parameter
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore import context
from .._c_expression import Primitive_, real_run_op, prim_type
from .._c_expression import signature_rw as sig_rw
from .._c_expression import signature_kind as sig_kind
from .._c_expression import signature_dtype as sig_dtype


class Primitive(Primitive_):
    """
    Primitive is the base class of primitives in python.

    Args:
        name (str): Name for the current Primitive.

    Examples:
        >>> add = Primitive('add')
        >>>
        >>> # or work with prim_attr_register:
        >>> # init a Primitive class with attr1 and attr2
        >>> class Add(Primitive):
        >>>     @prim_attr_register
        >>>     def __init__(self, attr1, attr2):
        >>>         # check attr1 and attr2 or do some initializations
        >>> # init a Primitive obj with attr1=1 and attr2=2
        >>> add = Add(attr1=1, attr2=2)
    """
    _repr_ignore_list = ['input_names', 'output_names']

    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.init_attrs = {"name": name}
        self._update_parameter = False
        Primitive_.__init__(self, name, self)
        if hasattr(self.__class__, '__mindspore_signature__'):
            sig = self._fill_signature(self.__class__.__mindspore_signature__)
            self.set_signatures(sig)

    def _fill_signature(self, signatures):
        """fills signature."""
        signatures_new = []
        for signature in signatures:
            if isinstance(signature, sig_dtype):
                signatures_new.append(("argument", sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD,
                                       sig_kind.KIND_EMPTY_DEFAULT_VALUE, signature))
            else:
                if len(signature) < 3:
                    raise ValueError(f"[Internal Error]Signature for one parameter len must > 3, but {signature}")
                if len(signature) == 3:
                    signature += (sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T_EMPTY_DEFAULT_VALUE)
                if len(signature) == 4:
                    signature += (sig_dtype.T_EMPTY_DEFAULT_VALUE,)
                signatures_new.append(signature)
        return tuple(signatures_new)

    def _clone(self):
        """
        Deeply clones the primitive object.

        Calls the __init__() method with the same arguments. This method is called in parser if the
        flag self.__setattr_flag__ is True.
        """
        cloned = copy.deepcopy(self)
        init_params = inspect.getfullargspec(cloned.__init__.decorated_func).args[1:]
        init_args = {}
        for name in init_params:
            value = self.attrs[name]
            init_args[name] = value
        # __init__ should be called to construct cpp object.
        cloned.__init__(**init_args)
        for name in self.attrs:
            value = self.attrs[name]
            cloned.add_prim_attr(name, value)
        if hasattr(self, 'instance_name'):
            cloned.set_prim_instance_name(self.instance_name)
        return cloned

    def add_prim_attr(self, name, value):
        """
        Adds primitive attribute.

        Args:
            name (str): Attribute Name.
            value (Any): Attribute value.
        """
        self.__dict__[name] = value
        self.attrs[name] = value
        self.add_attr(name, value)
        return self

    def set_strategy(self, strategy):
        """
        Add strategies to primitive attribute.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy (tuple): Strategy describes the distributed parallel mode of the current primitive.
        """
        self.add_prim_attr("strategy", strategy)
        return self

    def set_prim_instance_name(self, instance_name):
        """
        Set instance name to primitive operator.

        Note:
            It will be called by default when user defines primitive operator.

        Args:
            instance_name (str): Instance name of primitive operator set by user.
        """
        self.set_instance_name(instance_name)
        self.instance_name = instance_name
        return self

    def __getattr__(self, item):
        if item == 'infer_dynamic_shape':
            return None
        if item in super().get_attr_dict():
            return super().get_attr_dict()[item]
        if item in self.attrs:
            return self.attrs[item]
        raise AttributeError(item)

    def check_elim(self, *args):
        """
        Check if certain inputs should go to the backend. Subclass in need should override this method.

        Args:
            *args(Primitive args): Same as arguments of current Primitive.

        Returns:
            A tuple consisting of two elements. The first element indicates whether we should filter out current
            arguments; the seconde element is the output if we need to filter out the arguments.
        """
        return (False, None)

    def __call__(self, *args):
        should_elim, output = self.check_elim(*args)
        if should_elim:
            return output
        return _run_op(self, self.name, args)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __deepcopy__(self, memo):
        return type(self)(**self.init_attrs)

    def __repr__(self):
        attr = ', '.join([f'{k}={self.attrs[k]}'for k in self.attrs if not k in Primitive._repr_ignore_list])
        info_str = f'Prim[{self.name}]'
        if attr:
            info_str += f'<{attr}>'
        return info_str

    def init_prim_io_names(self, inputs, outputs):
        """
        Initializes the name of inputs and outpus of Tensor or attributes.

        Args:
            inputs (list[str]): list of inputs names.
            outputs (list[str]): list of outputs names.
        """
        # for checking para names with kernel implementation
        self.add_prim_attr("input_names", inputs)
        # for checking output number with kernel implementation
        self.add_prim_attr("output_names", outputs)

    @property
    def update_parameter(self):
        """ Whether the primitive will update the value of parameter."""
        return self._update_parameter


class PrimitiveWithInfer(Primitive):
    """
    PrimitiveWithInfer is the base class of primitives in python defines functions for tracking inference in python.

    There are four method can be overide to define the infer logic of the primitive: __infer__(), infer_shape(),
    infer_dtype(), and infer_value(). If __infer__() is defined in primitive, the __infer__() has highest priority
    to be called. If __infer__() is not defined, infer_shape() and infer_dtype() can be defined to describe the infer
    logic of the shape and type. The infer_value() is used for constant propagation.

    Args:
        name (str): Name of the current Primitive.

    Examples:
        >>> # init a Primitive class with infer
        >>> class Add(PrimitiveWithInfer):
        >>>     @prim_attr_register
        >>>     def __init__(self):
        >>>         pass
        >>>
        >>>     def infer_shape(self, x, y):
        >>>         return x # output shape same as first input 'x'
        >>>
        >>>     def infer_dtype(self, x, y):
        >>>         return x # output type same as first input 'x'
        >>>
        >>> # init a Primitive obj
        >>> add = Add()
    """

    def __init__(self, name):
        Primitive.__init__(self, name)
        self.set_prim_type(prim_type.py_infer_shape)

    def _clone(self):
        """
        Deeply clones the primitive object.

        Calls the __init__() method with the same arguments. This method is called in parser if the
        flag self.__setattr_flag__ is True.
        """
        cloned_prim = Primitive._clone(self)
        return cloned_prim

    def infer_shape(self, *args):
        """
        Infer output shape based on input shape.

        Note:
            The shape of scalar is an empty tuple.

        Args:
            args (tuple(int)): shapes of input tensors.

        Return:
            `tuple(int)`, shapes of output tensors.
        """
        return None

    def infer_dtype(self, *args):
        """
        Infer output dtype based on input dtype.

        Args:
            args (:class:`mindspore.dtype`): data type of inputs.

        Return:
            :class:`mindspore.dtype`, data type of outputs.
        """
        return None

    def infer_value(self, *args):
        """
        Infer output value based on input value at compile time.

        Args:
            args (Any): value of inputs.

        Return:
            Value of outputs. Return `None`, the value can not be inferred at compile time in this case.
        """
        return None

    def __infer__(self, *args):
        """Infer shape, type, and value at the same time by using dictionary as arguments."""
        is_graph_mode = context.get_context("mode") == context.GRAPH_MODE
        fn_infer_dynamic_shape = getattr(self, 'infer_dynamic_shape', None)
        if is_graph_mode and fn_infer_dynamic_shape is not None:
            out = fn_infer_dynamic_shape(*args)
            tracks = ['dtype', 'value']
            for track in tracks:
                fn = getattr(self, 'infer_' + track)
                # fn may return None
                out[track] = fn(*(x[track] for x in args))
            return out

        tracks = ['dtype', 'shape', 'value']
        out = {}
        for track in tracks:
            fn = getattr(self, 'infer_' + track)
            # fn may return None
            out[track] = fn(*(x[track] for x in args))

        # in non-graph_mode, it is not necessary to infer min/max shape
        if not is_graph_mode:
            return out

        def get_specified_shape(elems, attr):
            has_specified_shape = False
            ret_vals = []
            for elem in elems:
                if attr in elem:
                    has_specified_shape = True
                    ret_vals.append(elem[attr])
                else:
                    ret_vals.append(elem['shape'])
            return has_specified_shape, tuple(ret_vals)

        has_min_shape, min_shapes = get_specified_shape(args, 'min_shape')
        has_max_shape, max_shapes = get_specified_shape(args, 'max_shape')
        if not (has_min_shape or has_max_shape):
            return out
        if has_min_shape and has_max_shape:
            fn_infer_shape = getattr(self, 'infer_shape')
            out['min_shape'] = fn_infer_shape(*min_shapes)
            out['max_shape'] = fn_infer_shape(*max_shapes)
            return out
        raise ValueError('Input args has invalid dynamic shape, args info: {args}')


def prim_attr_register(fn):
    """
    Primitive attributes register.

    Register the decorator of the built-in operator primitive '__init__'.
    The function will add all the parameters of '__init__' as operator attributes.

    Args:
        fn (function): __init__ function of primitive.

    Returns:
        function, original function.
    """
    def deco(self, *args, **kwargs):
        if isinstance(self, PrimitiveWithInfer):
            PrimitiveWithInfer.__init__(self, self.__class__.__name__)
        else:
            Primitive.__init__(self, self.__class__.__name__)
        bound_args = inspect.signature(fn).bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        arguments = bound_args.arguments
        del arguments['self']
        del self.init_attrs['name']
        for name in arguments:
            value = arguments[name]
            self.add_prim_attr(name, value)
            self.init_attrs[name] = value
        fn(self, *args, **kwargs)
    deco.decorated_func = fn
    return deco


def constexpr(fn=None, get_instance=True, name=None):
    """
    Make a PrimitiveWithInfer operator that can infer the value at compile time. We can use it to define a function to
    compute constant value using the constants in the constructor.

    Args:
        fn (function): A `fn` use as the infer_value of the output operator.
        get_instance (bool): If true, return the instance of operator, otherwise return the operator class.
        name (str): Defines the operator name. If `name` is None, use the function name as op name.

    Examples:
        >>> a = (1, 2)
        >>> # make an operator to calculate tuple len
        >>> @constexpr
        >>> def tuple_len(x):
        >>>     return len(x)
        >>> assert tuple_len(a) == 2
        >>>
        >>> # make a operator class to calculate tuple len
        >>> @constexpr(get_instance=False, name="TupleLen")
        >>> def tuple_len_class(x):
        >>>     return len(x)
        >>> assert tuple_len_class()(a) == 2
    """
    def deco(fn):
        class CompileOp(PrimitiveWithInfer):
            def __init__(self):
                op_name = name if name else fn.__name__
                PrimitiveWithInfer.__init__(self, op_name)
                self.set_is_const_value(True)

            def infer_value(self, *args):
                return fn(*args)
        if get_instance:
            return CompileOp()
        return CompileOp
    if fn is not None:
        return deco(fn)
    return deco


@_wrap_func
def _run_op(obj, op_name, args):
    """Single op execution function supported by ge in PyNative mode."""
    cast = tensor_operator_registry.get("cast")
    if op_name == "Cast" or obj.update_parameter:
        cast_args = args
    else:
        cast_args = list()
        for arg in args:
            if isinstance(arg, Parameter):
                if arg.cast_type:
                    cast_args.append(cast(arg, arg.cast_type))
                else:
                    cast_args.append(arg)
            else:
                cast_args.append(arg)
    output = real_run_op(obj, op_name, tuple(cast_args))
    if not output:
        raise RuntimeError("Pynative run op %s failed!" % op_name)
    if len(output) == 1:
        output = output[0]
    return output
