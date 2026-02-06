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

"""Define pijit context."""

import inspect
import sys
import types
import functools
import importlib.util
import mindspore
from mindspore import log as logger
from mindspore.graph.jit_config import JitConfig
from mindspore._c_expression import PreJit
from mindspore._c_expression import GraphExecutor_, jit_mode_pi_enable, jit_mode_pi_disable, pi_jit_set_context

_PY312_OR_LATER = sys.version_info >= (3, 12)
_PY_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _update_graph_executor_config(jit_config):
    """Update GraphExecutor jit_config"""
    if isinstance(jit_config, JitConfig):
        jit_config = jit_config.jit_config_dict
    if not isinstance(jit_config, dict):
        return
    valid_config = {}
    for k, v in jit_config.items():
        valid_config[str(k)] = str(v)
    GraphExecutor_.get_instance().set_jit_config(JitConfig(**valid_config).jit_config_dict)


class Unsupported(RuntimeError):
    """If using @jit(fullgraph=True), pijit will raise an Unsupported exception when encountering a graph break."""

    # pylint: disable=useless-super-delegation
    def __init__(self, msg: str):
        super().__init__(msg)


class PIJitCaptureContext:
    """
    Context manager for pijit graph capture
    """

    def __init__(self, fullgraph=False, jit_config=None):
        _update_graph_executor_config(jit_config)
        config = {'fullgraph': fullgraph}
        if isinstance(jit_config, JitConfig):
            config.update(jit_config.jit_config_dict)
        elif jit_config is not None:
            config.update(jit_config)

        disable_pijit = config.get('_disable_pijit', None)
        if disable_pijit is not None and not callable(disable_pijit):
            raise TypeError(f"The config '_disable_pijit' must be callable but got {disable_pijit}")

        self.config = config
        self.ret = None
        self.fn = None
        self._init_arg = iter([self.config])

        if not SKIP_RULES:
            return
        pi_jit_set_context(wrapper=self._wrapper(),
                           skip_files=_get_skip_files(),
                           skip_codes=SKIP_RULES["codes"])
        SKIP_RULES.clear()

    @staticmethod
    def _is_unsupported(fn):
        # generator, coroutine, awaitable and a function that return them is unsupported
        return inspect.isgeneratorfunction(fn) or inspect.iscoroutinefunction(fn) \
            or inspect.isasyncgenfunction(fn) or inspect.isawaitable(fn)

    def _wrapper(self):
        """
        Create and return the JIT wrapper for the original function.
        """
        def _fn(*args, **kwds):
            if _PY312_OR_LATER:
                raise Unsupported(
                    '@jit(capture_mode="bytecode") does not support Python 3.12+. '
                    f"Current Python version: {_PY_VERSION}"
                )
            PreJit(args, kwds)
            disable_pijit = self.config.get('_disable_pijit', None)
            if disable_pijit is not None and disable_pijit(args, kwds):
                # JIT is disabled for these inputs, call original function
                return self.fn(*args, **kwds)
            with self:
                self.ret = self.fn(*args, **kwds)
                return self.ret

        return _fn

    def __call__(self, fn):
        """
        :raises Unsupported: If using @jit(fullgraph=True), will raise exception when encountering a graph break.
        """
        if isinstance(fn, type) and issubclass(fn, mindspore.nn.Cell):
            fn.construct = self(fn.construct)
            return fn
        if isinstance(fn, mindspore.nn.Cell):
            return types.MethodType(self(fn.construct.__func__), fn)
        if isinstance(fn, types.MethodType):
            return types.MethodType(self(fn.__func__), fn.__self__)
        if not isinstance(fn, types.FunctionType) or self._is_unsupported(fn):
            logger.warning("unsupported function type" + str(fn))
            return fn

        if hasattr(fn, "__wrapped_by_jit__"):
            logger.warning(f"The fn {fn} should be wrapped by jit only once.")

        module = inspect.getmodule(fn.__code__)
        if module is not None and module.__name__.startswith("mindspore"):
            if fn.__code__.co_name != 'after_grad':
                # Use PIJit for mindspore api, please use PSJit
                return fn

        _fn = self._wrapper()
        if fn.__code__ is _fn.__code__:
            fn = fn.__closure__[0].cell_contents.fn
        self.fn = fn
        wrap_fn = functools.wraps(fn)(_fn)
        setattr(wrap_fn, "__wrapped_by_jit__", True)
        return wrap_fn

    def __enter__(self):
        pi_jit_set_context(self.fn, *self._init_arg)
        jit_mode_pi_enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pi_jit_set_context(None)
        jit_mode_pi_disable()


def _get_skip_files():
    """
    Get skip files by SKIP_RULES
    """

    def _filter(path: str):
        if path.endswith("__init__.py"):
            return path[0:-11]
        return path

    # not import these modules, only find it
    find = importlib.util.find_spec

    files = [*SKIP_RULES["skip_dirs"]]
    files += [_filter(find(m).origin) for m in SKIP_RULES["builtins"]]
    for i in SKIP_RULES["third_party"]:
        spec = find(i)
        if spec is None:
            continue
        files.append(_filter(spec.origin))

    return tuple(files)


# complete the skip list...
SKIP_RULES = {
    "skip_dirs": (
        "<frozen importlib",
        "<__array_function__ internals>",
        "<string>",
    ),
    "builtins": (
        "mindspore",  # not capture any function of mindspore unless it's called by user
        "abc",
        "ast",
        "codecs",
        "collections",
        "contextlib",
        "copy",
        "copyreg",
        "dataclasses",
        "enum",
        "functools",
        "glob",
        "importlib",
        "inspect",
        "linecache",
        "logging",
        "multiprocessing",
        "operator",
        "os",
        "posixpath",
        "random",
        "re",
        "selectors",
        "signal",
        "tempfile",
        "threading",
        "tokenize",
        "traceback",
        "types",
        "typing",
        "unittest",
        "weakref",
        "_collections_abc",
        "_weakrefset",
        # others...
        "sre_compile",
        "sre_parse",
        "genericpath",
    ),
    "third_party": (
        "numpy",
        "pandas",
        "sklearn",
        "tqdm",
        "tree",
    ),
    "codes": (),
}
