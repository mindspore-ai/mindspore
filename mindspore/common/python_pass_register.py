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
from mindspore.common.graph_pattern import Pattern
from mindspore._c_expression import PyPassManager_
from mindspore._c_expression import phase

class PyPassManager(PyPassManager_):
    r"""
    Used to registe and unregiste python passes which can be used to alter graphs.

    Args:
        pipeline_phase (phase): Specify the stage in which the pass will run in the pipeline. Default: phase.opt.
        run_only_once (bool): Specify whether or not to run pass only once. Default: False.
        multigraph (bool): Whether or not the pattern exists across graphs. Default: True.

    Raises:
        TypeError: If argument has invalid type.
    """
    def __init__(self, pipeline_phase=phase.opt, run_only_once=False, multi_graph=True):
        if not isinstance(pipeline_phase, phase):
            raise TypeError(f"Expecting phase, got : ({type(pipeline_phase)}){pipeline_phase}")
        if not isinstance(run_only_once, bool):
            raise TypeError(f"Expecting bool, got : ({type(run_only_once)}){run_only_once}")
        if not isinstance(multi_graph, bool):
            raise TypeError(f"Expecting bool, got : ({type(multi_graph)}){multi_graph}")
        PyPassManager_.__init__(self)
        self.phase_ = pipeline_phase
        self.run_only_once_ = run_only_once
        self.multi_graph_ = multi_graph

    def registe(self, py_pass):
        if not isfunction(py_pass):
            raise TypeError(f"Expecting function pass, got : ({type(py_pass)}){py_pass}")
        pattern, target = py_pass()
        pass_name = py_pass.__name__
        if not isinstance(pattern, Pattern):
            raise TypeError(f"Expecting pattern of Pattern type, got : ({type(pattern)}){pattern}")
        if not isinstance(target, Pattern):
            raise TypeError(f"Expecting target of Pattern type, got : ({type(target)}){target}")
        super().registe(pass_name, pattern, target, self.phase_, self.run_only_once_, self.multi_graph_)

    def unregiste(self, py_pass, pipeline_phase=phase.opt):
        if not isinstance(pipeline_phase, phase):
            raise TypeError(f"Expecting phase, got : ({type(pipeline_phase)}){pipeline_phase}")
        if isinstance(py_pass, str):
            super().unregiste(py_pass, pipeline_phase)
            return
        if isfunction(py_pass):
            super().unregiste(py_pass.__name__, pipeline_phase)
            return
        raise TypeError(f"Expecting py_pass to be string or function, got ({type(py_pass)}){py_pass}")

    def __call__(self, py_pass):
        self.registe(py_pass)
        return py_pass

def registe_pass(pipeline_phase=phase.opt, run_only_once=False, multi_graph=True):
    """
        Examples:
    >>> @registe_pass()
    >>> def toy_pass():
    >>>     def pattern():
    >>>         pass
    >>>     def target():
    >>>         pass
    """
    return PyPassManager(pipeline_phase, run_only_once, multi_graph)
