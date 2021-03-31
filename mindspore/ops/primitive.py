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
from mindspore import context
from .._c_expression import Primitive_, real_run_op, prim_type
from .._checkparam import Validator
from . import signature as sig


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
        ...     @prim_attr_register
        ...     def __init__(self, attr1, attr2):
        ...         '''init for add'''
        ...     # check attr1 and attr2 or do some initializations
        ...     # init a Primitive obj with attr1=1 and attr2=2
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
            out = self._fill_signature(self.__class__.__mindspore_signature__)
            self.set_signatures(out)

    def _fill_signature(self, signatures):
        """fills signature."""
        signatures_new = []
        for signature in signatures:
            if isinstance(signature, sig.Signature):
                signatures_new.append(signature)
            elif isinstance(signature, sig.sig_dtype):
                signatures_new.append(sig.make_sig(dtype=signature))
            else:
                if len(signature) < 3:
                    raise ValueError(f"[Internal Error]Signature for one parameter len must > 3, but {signature}")
                signatures_new.append(sig.make_sig(*signature))
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

    def del_prim_attr(self, name):
        """
        Del primitive attribute.

        Args:
            name (str): Attribute Name.
        """
        if name in self.__dict__ and name in self.attrs:
            del self.__dict__[name]
            del self.attrs[name]
            self.del_attr(name)
        return self

    def set_stage(self, stage):
        """
        Add stage id to primitive attribute.

        Note:
            It is valid only in semi auto parallel.
            In other parallel modes, please set it to be 0.

        Args:
            stage (int): The stage id for the current operation
        """
        self.add_prim_attr("stage", stage)
        return self

    def shard(self, strategy):
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
            args(Primitive args): Same as arguments of current Primitive.

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
        attr = ', '.join([f'{k}={self.attrs[k]}' for k in self.attrs if not k in Primitive._repr_ignore_list])
        info_str = f'Prim[{self.name}]'
        if attr:
            info_str += f'<{attr}>'
        return info_str

    def init_prim_io_names(self, inputs, outputs):
        """
        Initializes the name of inputs and outputs of Tensor or attributes.

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

    def recompute(self, mode=True):
        """
        Set the primitive recomputed. If a primitive set recomputed feeds into some backward nodes
        for computing gradient, rather than storing the intermediate activation computed in forward
        pass, we will recompute it in backward pass.

        Note:

            - If the computation involves something like randomization or global variable, the equivalence
              is not guaranteed currently.

        Args:
            mode (bool): Specifies whether the primitive is recomputed. Default: True.
        """
        if context.get_context("mode") == context.PYNATIVE_MODE:
            raise TypeError("Recompute is not supported in pynative mode currently.")
        Validator.check_bool(mode)
        self.add_prim_attr("recompute", mode)
        return self


class PrimitiveWithCheck(Primitive):
    """
    PrimitiveWithCheck is the base class of primitives in python defines functions for checking operator input arguments
    but used the infer method registered in c++ source codes.

    There are three methods can be override to define the check logic of the primitive: __check__(), check_shape(),
    check_dtype(). If __check__() is defined in primitive, the __check__() has highest priority to be called.
    If __check__() is not defined, check_shape() and check_dtype() can be defined to describe the check logic of
    the shape and type. Method infer_value() can also be defined (such as PrimitiveWithInfer) for constant propagation.

    Args:
        name (str): Name of the current Primitive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # init a Primitive class with check
        >>> class Flatten(PrimitiveWithCheck):
        ...     @prim_attr_register
        ...     def __init__(self):
        ...         pass
        ...     def check_shape(self, input_x):
        ...         validator.check_int(len(input_x), 1, Rel.GE, 'input_x rank', self.name)
        ...
        ...     def check_dtype(self, input_x):
        ...         validator.check_subclass("input_x", input_x, mstype.tensor, self.name)
        ...
        >>> # init a Primitive obj
        >>> add = Flatten()
    """

    def __init__(self, name):
        Primitive.__init__(self, name)
        self.set_prim_type(prim_type.py_infer_check)

    def _clone(self):
        """
        Deeply clones the primitive object.

        Calls the __init__() method with the same arguments. This method is called in parser if the
        flag self.__setattr_flag__ is True.
        """
        cloned_prim = Primitive._clone(self)
        return cloned_prim

    def check_shape(self, *args):
        """
        Check shapes of input args.

        Note:
            The shape of scalar is an empty tuple.

        Args:
            args (tuple(int)): shapes of input tensors.

        Return:
            None.
        """
        return None

    def check_dtype(self, *args):
        """
        Check data types of input args.

        Args:
            args (:class:`mindspore.dtype`): data type of inputs.

        Return:
            None.
        """
        return None

    def __check__(self, *args):
        """Check shape, type, and value at the same time by using dictionary as arguments."""
        tracks = ['dtype', 'shape']
        for track in tracks:
            fn = getattr(self, 'check_' + track)
            fn(*(x[track] for x in args))


class PrimitiveWithInfer(Primitive):
    """
    PrimitiveWithInfer is the base class of primitives in python and defines functions for tracking inference in python.

    There are four method can be override to define the infer logic of the primitive: __infer__(), infer_shape(),
    infer_dtype(), and infer_value(). If __infer__() is defined in primitive, the __infer__() has highest priority
    to be called. If __infer__() is not defined, infer_shape() and infer_dtype() can be defined to describe the infer
    logic of the shape and type. The infer_value() is used for constant propagation.

    Args:
        name (str): Name of the current Primitive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # init a Primitive class with infer
        >>> class Add(PrimitiveWithInfer):
        ...     @prim_attr_register
        ...     def __init__(self):
        ...         pass
        ...
        ...     def infer_shape(self, x, y):
        ...         return x # output shape same as first input 'x'
        ...
        ...     def infer_dtype(self, x, y):
        ...         return x # output type same as first input 'x'
        ...
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

        # output does not contain dynamic shape, no need to calculate min/max shape
        def has_dynamic_shape(shp):
            if isinstance(shp, int):
                return shp < 0
            if isinstance(shp, (list, tuple)):
                return any(has_dynamic_shape(e) for e in shp)
            return False

        if not has_dynamic_shape(out['shape']):
            return out

        # calculate min/max shape for output
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
        class_name = self.__class__.__name__
        if hasattr(self.__class__, "substitute_name"):
            class_name = self.__class__.substitute_name
        if isinstance(self, PrimitiveWithInfer):
            PrimitiveWithInfer.__init__(self, class_name)
        elif isinstance(self, PrimitiveWithCheck):
            PrimitiveWithCheck.__init__(self, class_name)
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
    Creates a PrimitiveWithInfer operator that can infer the value at compile time. We can use it to define a function
    to compute constant value using the constants in the constructor.

    Args:
        fn (function): A `fn` use as the infer_value of the output operator.
        get_instance (bool): If true, return the instance of operator, otherwise return the operator class.
        name (str): Defines the operator name. If `name` is None, use the function name as op name.

    Examples:
        >>> a = (1, 2)
        >>> # make an operator to calculate tuple len
        >>> @constexpr
        >>> def tuple_len(x):
        ...     return len(x)
        >>> assert tuple_len(a) == 2
        ...
        >>> # make an operator class to calculate tuple len
        >>> @constexpr(get_instance=False, name="TupleLen")
        >>> def tuple_len_class(x):
        ...     return len(x)
        >>> assert tuple_len_class()(a) == 2
    """

    def deco(fn):
        class CompileOp(PrimitiveWithInfer):
            def __init__(self):
                op_name = name if name else fn.__name__
                PrimitiveWithInfer.__init__(self, op_name)
                self.set_const_prim(True)

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
    output = real_run_op(obj, op_name, args)
    return output
