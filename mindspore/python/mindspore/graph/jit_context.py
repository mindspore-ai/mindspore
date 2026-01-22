# Copyright 2024 Huawei Technologies Co., Ltd
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

"""JIT Context for various JIT compile."""


class JitContext():
    """JIT context"""
    def __init__(self):
        self._compiled = False
        self._result = None
        self._phase = None
        self._args = None

    def run_op(self, prim, prim_res, *args):
        raise AttributeError("For 'JitContext', the method 'run_op' is not defined.")

    @property
    def compiled(self):
        return self._compiled

    @compiled.setter
    def compiled(self, value):
        if not isinstance(value, bool):
            raise TypeError(f"For 'JitContext', the property 'compiled' must be bool type, but got type {type(value)}.")
        self._compiled = value

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if not isinstance(value, str):
            raise TypeError(f"For 'JitContext', the property 'phase' must be str type, but got type {type(value)}.")
        self._phase = value

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value


_jit_context = None


def set_jit_context(new_jit_context):
    global _jit_context
    _jit_context = new_jit_context


def jit_context():
    return _jit_context
