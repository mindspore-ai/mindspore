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
"""Python pass register"""
from inspect import isfunction
from mindspore.graph_utils.graph_pattern import Pattern, NewParameter
from mindspore._c_expression import PyPassManager_


__all__ = [
    "registe_pass",
    "unregiste_pass",
    "gen_new_parameter",
    "cancel_new_parameter",
    "set_renorm",
    "set_reopt"
]
class PyPassManager(PyPassManager_):
    r"""
    Used to registe and unregiste python passes which can be used to alter graphs.

    Args:
        run_only_once (bool): Specify whether or not to run pass only once. Default: False.

    Raises:
        TypeError: If argument has invalid type.
    """
    def __init__(self, run_only_once=False):
        if not isinstance(run_only_once, bool):
            raise TypeError(f"Expect bool, got : ({type(run_only_once)}){run_only_once}")
        self.run_only_once_ = run_only_once
        PyPassManager_.__init__(self)

    def registe(self, py_pass):
        if not isfunction(py_pass):
            raise TypeError(f"Expect function pass, got : ({type(py_pass)}){py_pass}")
        pattern, target = py_pass()
        pass_name = py_pass.__name__
        if not isinstance(pattern, Pattern):
            raise TypeError(f"Expect pattern of Pattern type, got : ({type(pattern)}){pattern}")
        if not isinstance(target, Pattern):
            raise TypeError(f"Expect target of Pattern type, got : ({type(target)}){target}")
        super().registe(pass_name, pattern, target, self.run_only_once_)

    def unregiste(self, py_pass):
        if isinstance(py_pass, str):
            super().unregiste(py_pass)
            return
        if isfunction(py_pass):
            super().unregiste(py_pass.__name__)
            return
        raise TypeError(f"Expect py_pass to be string or function, got ({type(py_pass)}){py_pass}")

    def __call__(self, py_pass):
        self.registe(py_pass)
        return py_pass

    def gen_new_parameter(self, pattern):
        if not isinstance(pattern, NewParameter):
            raise TypeError(f"Expect pattern to be a NewParameter Pattern, got {pattern}")
        super().gen_new_parameter(pattern)

    def set_renorm(self, should_renorm):
        if not isinstance(should_renorm, bool):
            raise TypeError(f"Expect should_renorm to be a bool, got {should_renorm}")
        super().set_renorm(should_renorm)

    def set_reopt(self, do_reopt):
        if not isinstance(do_reopt, bool):
            raise TypeError(f"Expect do_reopt to be a bool, got {do_reopt}")
        super().set_reopt(do_reopt)

def registe_pass(run_only_once=False):
    """
    Registe python pass to specified pipeline phase which would be used in compilation.

    Args:
        run_only_once(bool): Run this pass only once if set true. Otherwise run the pass until converge. Default: False.

    Returns:
        This function should be used as a decorator, return the decoratorated pass function.

    Examples:
        >>> from mindspore.graph_utils.graph_pattern import IsPrimTypeOf
        >>> @registe_pass()
        >>> def toy_pass():
        >>>     pattern = IsPrimTypeOf("ReLU")
        >>>     target = IsPrimTypeOf("ReLU6")
        >>>     return pattern, target
    """
    return PyPassManager(run_only_once)

def unregiste_pass(py_pass):
    """
    Unregiste python pass.

    Args:
        py_pass(Union(str, function)): target python pass to unregiste.
    """
    ppm = PyPassManager()
    ppm.unregiste(py_pass)

def gen_new_parameter(pattern):
    """
    Generate specified parameter every time a network gets compiled.

    NOTE:
        In this way, every pass uses this pattern would be using the same Parameter. If use NewParameter without
        gen_new_parameter, every pass match would build a new Parameter.
        This would registe a pass to add new parameter in the compilation pipeline, so later compilation would
        ALSO add this parameter unless the pass is unregisted. To unregiste this pass, call
        cancel_new_parameter(pattern)

    Args:
        pattern (NewParameter): NewParameter type, could be used to build nested patterns across multiple passes
            after gen_new_parameter.

    Raises:
        TypeError: If argument has invalid type.

    Examples:
        >>> from mindspore.graph_utils.graph_pattern import NewParameter
        >>> abc = NewParameter("abc")
        >>> gen_new_parameter(abc)
    """
    ppm = PyPassManager()
    ppm.gen_new_parameter(pattern)

def cancel_new_parameter(pattern):
    """
    Use with gen_new_parameter to unregiste gen_new_parameter pass.

    Args:
        pattern (NewParameter): NewParameter type, cancel the pass which would add new parameter as this pattern
            describes.

    Examples:
        >>> from mindspore.graph_utils.graph_pattern import NewParameter
        >>> abc = NewParameter("abc")
        >>> gen_new_parameter(abs)
        >>> # some compilations
        >>> cancel_new_parameter(abc)
    """
    if not isinstance(pattern, NewParameter):
        raise TypeError(f"Expect pattern to be a NewParameter Pattern, got {pattern}")
    ppm = PyPassManager()
    ppm.unregiste(pattern.para_name)

def set_renorm(should_renorm):
    """
    Set whether or not to do renormalization after modified graph in python pass(es).

    Args:
        should_renorm(bool): whether or not to do renormalization after modified graph in python pass(es).

    NOTE:
        This interface is mainly intended for testing modifying graph without worrying about its validity. Turn off
        renormalization may BREAK the network.
    """
    ppm = PyPassManager()
    ppm.set_renorm(should_renorm)

def set_reopt(do_reopt):
    """
    Set whether or not to do optimization after modified graph in python pass(es).

    Args:
        do_reopt(bool): whether or not to do optimization after modified graph in python pass(es).

    NOTE:
        This interface is mainly intended for testing modifying graph without worrying about its validity. Turn off
        renormalization may BREAK the network.
    """
    ppm = PyPassManager()
    ppm.set_reopt(do_reopt)
