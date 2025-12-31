# Copyright 2021-2025 Huawei Technologies Co., Ltd
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

"""Custom operator"""
from __future__ import absolute_import
import json
import os
import re
import sys
import ast
import hashlib
import stat
import inspect
import importlib
import platform
import subprocess
import shutil
import threading
from typing import Any, Callable, get_type_hints
import numpy as np
import mindspore as ms
from mindspore._c_expression import Oplib, typing
from mindspore._c_expression import pyboost_custom_ext
from mindspore._c_expression import PyFuncOpDefBuilder
from mindspore import context
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import DataType
from mindspore import log as logger
from mindspore import ops
from mindspore.communication.management import get_rank, GlobalComm
from mindspore import _checkparam as validator
from mindspore.ops.primitive import Primitive, prim_arg_register
from mindspore.ops import signature as sig
from ._ms_kernel import determine_variable_usage
from ._custom_grad import autodiff_bprop
from ._pyfunc_registry import add_pyfunc
from ._custom_ops_utils import ExtensionBuilder, CustomCodeGenerator, CustomInfoGenerator

if platform.system() != "Windows":
    import fcntl

KEY_ATTR = "attr"
KEY_NAME = "name"
INPUT_NAMES = "input_names"
ATTR_NAMES = "attr_names"
AUTO_DIFF = "autodiff"
IMPLY_TYPE = "imply_type"
FUSION_TYPE = "fusion_type"
MS_KERNEL_FLAG = "ms_kernel_flag"
AKG = "AKG"
TBE = "TBE"
CUDA = "CUDA"
AICORE = "AiCore"
CPU = "CPU"
GPU = "GPU"
ASCEND = "Ascend"
HYBRID_TYPE = "hybrid"
OP_NAME = "op_name"


def _get_cache_path():
    """
    Automatically get the path to kernel_meta

    Returns:
        str, the path to the dir of kernel_meta.
    """
    cache_path = os.getenv('MS_COMPILER_CACHE_PATH')
    if cache_path is None:
        cache_path = "./custom_kernel_meta/"
    elif cache_path[-1] != "/":
        cache_path = cache_path + "/"

    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)

    # for distributed case, we create folders separately to avoid conflict
    if GlobalComm.INITED:
        cache_path = os.path.join(cache_path, "rank_" + str(get_rank()), "")
        os.makedirs(cache_path, exist_ok=True)

    return cache_path


def _compile_aot(file):
    """
    Automatically compile the source file for custom aot

    Args:
        file (str): The path to the source file.

    Returns:
        str, the path to the compiled library.
    """
    cache_path = _get_cache_path()

    res_path = importlib.util.find_spec("mindspore").origin
    find_pos = res_path.find("__init__.py")
    if find_pos == -1:
        raise RuntimeError(
            "Find module mindspore __init__.py file failed!")
    include_file = "-I{}include/api/".format(res_path[:find_pos])

    file_name = file.split('/')[-1]
    func_path = cache_path + file_name + ".so"
    include_file = "{} -I{}".format(include_file, file[:file.rindex('/')])

    if context.get_context("device_target") == "Ascend":
        ascend_cann_path = os.getenv("ASCEND_OPP_PATH").split('opp')[0]
        ascend_include = os.path.join(ascend_cann_path, "include")
        include_file = "{} -I{}".format(include_file, ascend_include)

    include_file = include_file.split(" ")
    if func_path not in Custom.compiled_bin:
        Custom.compiled_bin.append(func_path)

        if file.endswith("cpp") or file.endswith("cc"):
            cmd = ["g++", "-std=c++17", "--shared", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=0"]
            cmd += include_file
            cmd += ["-o", func_path, file]
        elif file.endswith("cu"):
            cmd = ["nvcc"]
            cmd += ["--shared", "-Xcompiler", "-fPIC", "-O3", "-gencode", "arch=compute_70, code=sm_70"]
            cmd += ["--use_fast_math", "--expt-relaxed-constexpr"]
            cmd += ["-D_GLIBCXX_USE_CXX11_ABI=0"]

            def _get_cuda_bare_metal_version():
                raw_output = subprocess.check_output(["nvcc", "-V"],
                                                     universal_newlines=True)
                output = raw_output.split()
                release_idx = output.index("release") + 1
                release = output[release_idx].split(".")
                version_idx = release_idx + 1
                version = output[version_idx].split(".")
                version_middle = version[1] if len(version) > 1 else 0
                version_minor = version[2] if len(version) > 2 else 0

                return int(release[0]), int(version_middle), int(version_minor)

            v_major, v_mid, v_minor = _get_cuda_bare_metal_version()
            if v_major >= 11:
                cmd += ["-gencode", "arch=compute_80,code=sm_80", "--expt-extended-lambda"]
            elif v_major == 10 and not (v_mid >= 1 and v_minor >= 168):
                logger.warning("The current version of nvcc, V{}.{}.{},  might have unfixed issues with std string, "
                               "which will lead to errors in aot custom op with attrs."
                               "The version higher than V10.1.168 is recommended".format(v_major, v_mid, v_minor))
            cmd += include_file
            cmd += ["-o", func_path, file]
        else:
            raise ValueError("The source file must be a cc/cpp/cu file, but get: {}".format(file))

        with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, shell=False) as proc:
            out, _ = proc.communicate(timeout=30)

        if proc.returncode != 0:
            msg = "Compilation error in compiling {}:\n".format(file)
            msg += out.decode('utf-8')
            raise RuntimeError(msg)

    return func_path


class _CustomExt(ops.PrimitiveWithInfer):
    """
    `Custom` primitive is used for PyBoost.
    """

    def __init__(self, func, out_shape=None, out_dtype=None, bprop=None):
        super().__init__("CustomExt")
        self.func = func
        self.out_shape = out_shape
        self.out_dtype = out_dtype
        self.bprop = bprop

    def __infer__(self, *args):
        if callable(self.out_shape):
            infer_shape = self.out_shape(*(x["shape"] for x in args))
        else:
            infer_shape = self.out_shape

        if callable(self.out_dtype):
            infer_dtype = self.out_dtype(*(x["dtype"] for x in args))
        else:
            infer_dtype = self.out_dtype

        infer_value = None
        if infer_shape is None:
            logger.debug("'out_shape' is None. Add a placeholder instead. "
                         "A CPP version of infer shape function is required "
                         "in this case.")
            infer_shape = (1,)
        if infer_dtype is None:
            logger.debug("'out_dtype' is None. Add a placeholder instead. "
                         "A CPP version of infer type function is required "
                         "in this case.")
            infer_dtype = ms.float16

        # after all automatic infer information fulfillment, throw error if infer_shape/infer_dtype is still None
        if not isinstance(infer_shape, (tuple, list)):
            raise TypeError("'out_shape' must be one of [tuple, list, function], but got {}".format(type(infer_shape)))

        if not isinstance(infer_dtype, (typing.Type, tuple, list)):
            raise TypeError("'out_dtype' must be one of [mindspore.dtype, tuple, list, function], but got {}"
                            .format(type(infer_dtype)))

        out = {
            "shape": infer_shape,
            "dtype": infer_dtype,
            "value": infer_value,
        }
        return out

    def get_bprop(self):
        """return back propagation function"""
        return self.bprop


class Custom(ops.PrimitiveWithInfer):
    r"""
    `Custom` primitive is used for user defined operators and is to enhance the expressive ability of built-in
    primitives. You can construct a `Custom` object with a predefined function, which describes the computation
    logic of a user defined operator. You can also construct another `Custom` object with another predefined
    function if needed. Then these `Custom` objects can be directly used in neural networks.
    Detailed description and introduction of user-defined operators, including correct writing of parameters,
    please refer to `Custom Operators Tutorial
    <https://www.mindspore.cn/tutorials/en/master/custom_program/op_custom.html>`_ .

    .. warning::
        This is an experimental API that is subject to change.

    .. note::
        The supported platforms are determined by the input `func_type`. The supported platforms are as follows:

        - "aot": supports ["GPU", "CPU", "Ascend"].
        - "pyfunc": supports ["CPU"].

    Args:
        func (Union[function, str]):

            - function: If func is of function type, then func should be a Python function which describes the
              computation logic of a user defined operator.

            - str: If func is of str type, then str should be a path of file along with a function name.
              This could be used when func_type is "aot".

              for "aot":

              a) GPU/CPU platform.
              "aot" means ahead of time, in which case Custom directly launches user defined "xxx.so" file as an
              operator. Users need to compile a handwriting "xxx.cu"/"xxx.cc" file into "xxx.so" ahead of time,
              and offer the path of the file along with a function name.

              - "xxx.so" file generation:

                1) GPU Platform: Given user defined "xxx.cu" file (ex. "{path}/add.cu"), use nvcc command to compile
                it.(ex. "nvcc --shared -Xcompiler -fPIC -o add.so add.cu")

                2) CPU Platform: Given user defined "xxx.cc" file (ex. "{path}/add.cc"), use g++/gcc command to
                compile it.(ex. "g++ --shared -fPIC  -o add.so add.cc")

              - Define a "xxx.cc"/"xxx.cu" file:

                "aot" is a cross-platform identity. The functions defined in "xxx.cc" or "xxx.cu" share
                the same args. Typically, the function should be as:

                .. code-block::

                    int func(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                            void *stream, void *extra)

                Parameters:

                - nparam(int): total number of inputs plus outputs; suppose the operator has 2 inputs and 3 outputs,
                  then nparam=5
                - params(void \*\*): a pointer to the array of inputs and outputs' pointer; the pointer type of
                  inputs and outputs is void \* ; suppose the operator has 2 inputs and 3 outputs, then the first
                  input's pointer is params[0] and the second output's pointer is params[3]
                - ndims(int \*): a pointer to the array of inputs and outputs' dimension num; suppose params[i] is a
                  1024x1024 tensor and params[j] is a 77x83x4 tensor, then ndims[i]=2, ndims[j]=3.
                - shapes(int64_t \*\*): a pointer to the array of inputs and outputs' shapes(int64_t \*); the ith
                  input's jth dimension's size is shapes[i][j](0<=j<ndims[i]); suppose params[i] is a 2x3 tensor and
                  params[j] is a 3x3x4 tensor, then shapes[i][0]=2, shapes[j][2]=4.
                - dtypes(const char \*\*): a pointer to the array of inputs and outputs' types(const char \*);
                  (ex. "float32", "float16", "float", "float64", "int", "int8", "int16", "int32", "int64", "uint",
                  "uint8", "uint16", "uint32", "uint64", "bool")
                - stream(void \*): stream pointer, only used in cuda file
                - extra(void \*): used for further extension

                Return Value(int):

                - 0: MindSpore will continue to run if this aot kernel is successfully executed
                - others: MindSpore will raise exception and exit

                Examples: see details in tests/st/ops/graph_kernel/custom/aot_test_files/

              - Use it in Custom:

                .. code-block::

                    Custom(func="{dir_path}/{file_name}:{func_name}",...)
                    (ex. Custom(func="./reorganize.so:CustomReorganize", out_shape=[1], out_dtype=mstype.float32,
                    "aot"))

              b) Ascend platform.
              Before using Custom operators on the Ascend platform, users must first develop custom operators
              based on Ascend C and compile them. The complete development and usage process can refer to the
              tutorial `AOT-Type Custom Operators(Ascend)
              <https://www.mindspore.cn/tutorials/en/master/custom_program/operation/op_custom_ascendc.html>`_.
              By passing the name of the operator through the input parameter `func`, there are two usage methods
              based on the implementation of the infer function:

              - Python infer: If the operator's infer function is implemented in Python, that is, the infer shape
                function is passed through the `out_shape` parameter, and the infer type is passed throuht the
                `out_dtype`, then the `func` should be specified as the operator name, for example,
                `func="CustomName"`.
              - C++ infer: If the operator's infer function is implemented through C++, then pass the path of the
                infer function implementation file in `func` and separate the operator name with `:`,
                for example: `func="add_custom_infer.cc:AddCustom"` .

        out_shape (Union[function, list, tuple], optional): The output shape infer function or the value of output
            shape of `func`. Default: ``None`` .

            If func has single output, then the value of output shape is a list or tuple of int.

            If func has multiple outputs, then the value of output shape is a tuple, each item represents the shape
            of each output.

            The input can be None only when the func_type input is "hybrid". In this case, the automatic infer
            shape mechanic will be enabled.

        out_dtype (Union[function, :class:`mindspore.dtype`, tuple[:class:`mindspore.dtype`]], optional): The output
            data type infer function or the value of output data type of `func`. Default: ``None`` .

            If func has single output, then the value of output shape is a `mindspore.dtype`.

            If func has multiple outputs, then the value of output shape is a tuple of `mindspore.dtype`, each item
            represents the data type of each output.

            The input can be None only when the func_type input is "hybrid". In this case, the automatic infer
            value mechanic will be enabled.

        func_type (str, optional): The implementation type of `func`, should be one of
            [ ``"aot"`` , ``"pyfunc"``]. Default: ``"pyfunc"``.

        bprop (function, optional): The back propagation function of `func`. Default: ``None`` .
        reg_info (Union[str, dict, list, tuple], optional): Represents the registration information(reg info) of `func`
            with json format of type str or dict. The reg info specifies supported data types and formats of inputs and
            outputs, attributes and target of `func`. Default: ``None`` .

            If reg info is a list or tuple, then each item should be with json format of type str or dict, which
            represents the registration information of `func` in a specific target. You need to invoke `CustomRegOp`
            or the subclass of `RegOp` to generate the reg info for `func`. Then you can invoke
            `custom_info_register` to bind the reg info to `func` or just pass the reg info to `reg_info` parameter.
            The `reg_info` parameter takes higher priority than `custom_info_register` and the reg info in a
            specific target will be registered only once.

            If reg info is not set, then we will infer the data types and formats from the inputs of `Custom` operator.

            Please note that, if `func_type` is "tbe" or the `func` only supports some specified data types and formats,
            or it has attribute inputs, then you should set the reg info for `func`.

    Inputs:
        - **input** (Union(tuple, list)) - The input tuple or list is made up of multiple tensors, and attributes
          value(optional).

    Outputs:
        Tensor or tuple[Tensor], execution results.

    Raises:
        TypeError: If the type of `func` is invalid or the type of register information for `func` is invalid.
        ValueError: If `func_type` is invalid.
        ValueError: If the register information is invalid, including the target is not supported, the input numbers
            or the attributes of `func` differs in different targets.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore.ops import CustomRegOp, custom_info_register, DataType, kernel
        >>> from mindspore import dtype as mstype
        >>> from mindspore.nn import Cell
        >>> input_x = Tensor(np.ones([16, 16]).astype(np.float32))
        >>> input_y = Tensor(np.ones([16, 16]).astype(np.float32))
        >>>
        >>> # Example, func_type = "pyfunc"
        >>> def func_pyfunc(x1, x2):
        ...     return x1 + x2
        >>>
        >>> test_pyfunc = ops.Custom(func_pyfunc, lambda x, _: x, lambda x, _: x, "pyfunc")
        >>> output = test_pyfunc(input_x, input_y)
        >>> print(output.shape)
        (16, 16)
    """

    registered_func = {}
    attr_dict = {}  # Save input_names and attr_names for func.
    compiled_bin = []  # Save names for compiled bin.
    tbe_path_checked = []  # Save paths for tbe functions which is safe to be imported as module.
    tbe_path_failed = []  # Save paths for tbe functions which fail to be imported as module.
    op_path_in_cache = []  # Save paths for op functions created in the cached.
    custom_aot_warning = True  # Flag to enable warnings about custom aot path white list

    def __init__(self, func, out_shape=None, out_dtype=None, func_type="pyfunc", bprop=None, reg_info=None):
        super().__init__("Custom")

        self.supported_targets = [ASCEND, GPU, CPU]
        self.supported_func_type = ["hybrid", "akg", "tbe", "aicpu", "aot", "pyfunc"]
        self.log_prefix = "For '{}', 'func_type': {}, 'func': {}".format(self.name, func_type, func)
        self.func = func
        self.func_type = func_type
        self.func_name = ""
        self.uniq_name = ""
        self.imply_path = ""
        self.func_source_str = ""
        self._func_compile_attrs = {}
        self._is_ms_kernel = False
        self.out_shape = out_shape
        self.out_dtype = out_dtype
        self.reg_info = reg_info
        self.is_ascend_c = (context.get_context("device_target") == "Ascend" and self.func_type == "aot")

        self._check_platform()
        self._check_func()
        self._generate_reg_info()
        self._update_func_info(self.reg_info)
        self.add_prim_attr("func_name", self.func_name)
        self.add_prim_attr("uniq_name", self.uniq_name)
        if self.func_type == HYBRID_TYPE:
            self.add_prim_attr("func_compile_attrs", self._func_compile_attrs)

        self.add_prim_attr("imply_path", self.imply_path)
        if self.func_type == "pyfunc":
            func_id = id(self.func)
            add_pyfunc(func_id, self.func)
            self.add_prim_attr("fn_id", func_id)

        self._set_infer_flag()
        self._set_multi_output_flag()

        self.bprop = bprop
        self.fake_output = False
        self.single_scalar_output = False
        if self.func_type == "pyfunc":
            if not self.out_dtype:
                self.fake_output = True
            elif not self.out_shape:
                self.single_scalar_output = True
            self.add_prim_attr("fake_output", self.fake_output)
            self.add_prim_attr("single_scalar_output", self.single_scalar_output)

        # Register info
        self._register_info(self.reg_info)

        if func_type == "akg":
            self._set_akg_kernel_type()

        if not self.bprop and self.func_type == HYBRID_TYPE:
            self._hybrid_autodiff(func_type)

        self.add_prim_attr("func_type", self.func_type)
        self._update_attr()

        if self.is_ascend_c:
            self.custom_pyboost = _CustomExt(self.func, self.out_shape, self.out_dtype, self.bprop)
            for key, value in super().get_attr_dict().items():
                self.custom_pyboost.add_prim_attr(key, value)
        self._generate_get_workspace_size_func()

    def _set_infer_flag(self):
        """set cpp infer attr"""
        if self.out_shape is None and self.func_type == "aot":
            self.add_prim_attr("cpp_infer_shape", True)
        if self.out_dtype is None and self.func_type == "aot":
            self.add_prim_attr("cpp_infer_type", True)

    def _set_multi_output_flag(self):
        outputs = self.reg_info.get("outputs", []) if self.reg_info else []
        self.multi_output = len(outputs) > 1 or (len(outputs) == 1 and outputs[0].get("paramType") == "dynamic")
        self.add_prim_attr("multi_output", self.multi_output)

    def __infer__(self, *args):
        if callable(self.out_shape):
            infer_shape = self.out_shape(*(x["shape"] for x in args))
        else:
            infer_shape = self.out_shape

        if callable(self.out_dtype):
            infer_dtype = self.out_dtype(*(x["dtype"] for x in args))
        else:
            infer_dtype = self.out_dtype

        infer_value = None

        # deal with the case of ms script
        # enable auto infer function if any infer information is missing
        if self._is_ms_kernel and (infer_dtype is None or infer_shape is None):
            logger.info("{}, 'out_shape' or 'out_dtype' is None, infer the output shape and output dtype "
                        "automatically. There might be some Python RuntimeWarning but it wouldn't influence the "
                        "result.".format(self.log_prefix))

            auto_infer_result = self._auto_infer(*args)

            # use automatically inferred shape/dtype if the input infer values are null
            infer_shape = auto_infer_result[0] if infer_shape is None else infer_shape
            infer_dtype = auto_infer_result[1] if infer_dtype is None else infer_dtype
            infer_value = auto_infer_result[2]

        # deal with case that the custom op is of type pyfunc with empty output
        if self.func_type == "pyfunc":
            if infer_shape == ():
                logger.warning("{}, 'out_shape' is an empty tuple. Add a placeholder instead. "
                               "Not recommend to use it as it could be any uninitialized data.".format(self.log_prefix))
                infer_shape = (1,)
            if infer_dtype == ():
                logger.warning("{}, 'out_dtype' is an empty tuple. Add a placeholder instead. "
                               "Not recommend to use it as it could be any uninitialized data.".format(self.log_prefix))
                infer_dtype = mstype.int32
        if self.func_type == "aot":
            if infer_shape is None:
                logger.debug("{}, 'out_shape' is None. Add a placeholder instead. "
                             "A CPP version of infer shape function is required "
                             "in this case.".format(self.log_prefix))
                infer_shape = (1,)
            if infer_dtype is None:
                logger.debug("{}, 'out_dtype' is None. Add a placeholder instead. "
                             "A CPP version of infer type function is required "
                             "in this case.".format(self.log_prefix))
                infer_dtype = ms.float16
        # after all automatic infer information fulfillment, throw error if infer_shape/infer_dtype is still None
        if not isinstance(infer_shape, (tuple, list)):
            raise TypeError("{}, 'out_shape' must be one of [tuple, list, function], but got {}"
                            .format(self.log_prefix, type(infer_shape)))

        if not isinstance(infer_dtype, (typing.Type, tuple, list)):
            raise TypeError("{}, 'out_dtype' must be one of [mindspore.dtype, tuple, list, function], but got {}"
                            .format(self.log_prefix, type(infer_dtype)))

        out = {
            "shape": infer_shape,
            "dtype": infer_dtype,
            "value": infer_value,
        }
        return out

    def get_bprop(self):
        """return back propagation function"""
        return self.bprop

    def _set_akg_kernel_type(self):
        self.add_prim_attr('func_source_str', self.func_source_str)
        if "ir_builder" in self.func_source_str:
            self.func_type = "ir_builder"
        elif "compute" in self.func_source_str:
            self.func_type = "tvm_compute"
        else:
            self.func_type = HYBRID_TYPE
            self._hybrid_func_analyser()

    def _check_aot_func(self):
        """Check the source code and bin lib for aot type custom op"""
        if not isinstance(self.func, str):
            raise TypeError("{}, 'func' must be of type str, but got {}".format(
                self.log_prefix, type(self.func)))
        file_name_list = self.func.split(":")
        if len(file_name_list) != 2:
            if callable(self.out_shape):
                return
            raise TypeError(
                "{}, 'func' should be like 'file_name:func_name', but got {}".format(
                    self.log_prefix, self.func))
        file_path = os.path.realpath(file_name_list[0])
        if os.environ.get('MS_CUSTOM_AOT_WHITE_LIST') is None:
            if Custom.custom_aot_warning:
                logger.info("{}, no white list is set and it might cause problems. "
                            "Set the legal path of the file in MS_CUSTOM_AOT_WHITE_LIST"
                            .format(self.log_prefix))
                Custom.custom_aot_warning = False
        else:
            legal_path = os.path.realpath(os.environ.get('MS_CUSTOM_AOT_WHITE_LIST'))
            if legal_path not in file_path:
                raise TypeError(
                    "{}, the legal path for the file is {}, but the file is {}".format(
                        self.log_prefix, legal_path, file_path))
        if file_path.endswith(("cu", "cpp", "cc")):
            file_path = _compile_aot(file_path)
            self.func = file_path + ":" + file_name_list[1]

    def _check_platform(self):
        """check the platform"""
        if platform.system() != 'Linux':
            raise Exception("Custom op only supported on Linux platform currently.")

    def _check_func(self):
        """Check the validity of func_type and type of func"""
        if self.func_type not in self.supported_func_type:
            raise ValueError("{}, 'func_type' must be one of {}, but got {}"
                             .format(self.log_prefix, self.supported_func_type, self.func_type))
        if self.func_type == "aot":
            self._check_aot_func()

        elif self.func_type == HYBRID_TYPE:
            if not hasattr(self.func, MS_KERNEL_FLAG):
                raise TypeError("{}, 'func' must be a function decorated by kernel".format(self.log_prefix))
            self._is_ms_kernel = True
            self._func_compile_attrs = getattr(self.func, "compile_attrs", {})
        elif self.func_type == "akg":
            if hasattr(self.func, MS_KERNEL_FLAG):
                logger.warning("{}. To have a better user experience, the mode hybrid is suggested "
                               "for the input function with decorator @kernel. "
                               "To enable this mode, set the 'func_type' to be \"hybrid\"".format(self.log_prefix))
        elif self.func_type == "pyfunc":
            if hasattr(self.func, MS_KERNEL_FLAG):
                logger.warning("{}. Now you are using the function with decorator @kernel in the mode pyfunc. "
                               "The kernel will be executed as a native python function, which might lead to "
                               "low efficiency. To accelerate the kernel, set the 'func_type' to be \"hybrid\""
                               .format(self.log_prefix))
        elif self.func_type == "tbe":
            if not callable(self.func):
                raise TypeError("{}, 'func' must be of type function, but got {}"
                                .format(self.log_prefix, type(self.func)))

    def _update_func_imply_path(self):
        """Update op_imply_path of func"""
        file_path = os.path.realpath(inspect.getfile(self.func))

        if not self.func_type == "tbe":
            # Custom ops with type other than tbe doesn't need to import from the path
            # use the file path directly
            return file_path
        # For the custom op of type tbe, the kernel compiler will import the module from file path.
        # we will try import in the initialization,
        if file_path in Custom.tbe_path_checked:
            logger.info("The file of {} has already been checked good to be imported.".format(self.func_name))
            return file_path

        if file_path not in Custom.tbe_path_failed:
            # As a single file might include multiply functions
            # we will not try the file path which already failed in previous trials
            try:
                mod_spec = importlib.util.spec_from_file_location(
                    self.func_name, file_path)
                custom_mod = importlib.util.module_from_spec(mod_spec)
                mod_spec.loader.exec_module(custom_mod)
            except (ImportError, RecursionError):
                Custom.tbe_path_failed.append(file_path)
            else:
                Custom.tbe_path_checked.append(file_path)
                return file_path

        # Create a new file for each tbe function
        op_imply_path = os.path.realpath(_get_cache_path() + self.func_name + ".py")
        if op_imply_path in Custom.op_path_in_cache:
            logger.info("The new file of {} has already been created.".format(self.func_name))
            return op_imply_path

        logger.warning("Fail to import the original source file. Create a new source file for {}. "
                       "The new file will not include the dependency for the op function. "
                       "Check the definition of the function {} "
                       "in the file: {}".format(self.func_name, self.func_name, op_imply_path))

        Custom.op_path_in_cache.append(op_imply_path)

        if os.path.exists(op_imply_path):
            try:
                os.remove(op_imply_path)
            except FileNotFoundError:
                logger.warning("Fail to remove the existing file. Check the definition of the function {} "
                               "in the file: {}".format(self.func_name, op_imply_path))

        with open(op_imply_path, 'at', encoding='utf-8') as file:
            if platform.system() != "Windows":
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            file.seek(0, 2)
            if file.tell() == 0:
                file.write(self.func_source_str)
        os.chmod(op_imply_path, stat.S_IRUSR | stat.S_IWUSR)
        return op_imply_path

    def _update_func_info(self, reg_info):
        """Update information of func"""
        if callable(self.func):
            # For the func_type other then hybrid, get the original function if func is decorated
            if "__wrapped__" in self.func.__dict__ and self.func_type not in ["hybrid", "pyfunc"]:
                self.func = self.func.__dict__["__wrapped__"]
            # func name
            self.func_name = self.func.__name__

            # source code of func, not include the decorator before def
            self.func_source_str = inspect.getsource(self.func)
            index = self.func_source_str.find("def ")
            if index != -1:
                self.func_source_str = self.func_source_str[index:]

            # update path of func for TBE type of custom op
            self.imply_path = self._update_func_imply_path()
            if self._is_ms_kernel:
                # static check for the Hybrid DSL in hybrid
                root = ast.parse(self.func_source_str)
                inplace_assign_output = determine_variable_usage(root, self.func_name)
                if inplace_assign_output:
                    self.add_prim_attr("inplace_assign_output",
                                       " ".join((str(j)
                                                 for i in inplace_assign_output
                                                 for j in i)))
                self.add_prim_attr('func_source_str', self.func_source_str)

            # unique func name
            sha256 = hashlib.sha256()
            sha256.update(self.imply_path.encode("utf-8"))
            sha256.update(self.func_source_str.encode("utf-8"))
            hash_str = sha256.hexdigest()
            self.uniq_name = self.name + "_" + self.func_name + "_" + hash_str
        elif isinstance(self.func, str):
            # func name
            self.func_name = self.func
            # uniq func name
            prefix = self.name
            if reg_info is None:
                reg_info = {}
            reg_info_list = self._get_expanded_list(reg_info)
            for reg_info_item in reg_info_list:
                if not isinstance(reg_info_item, (str, dict)):
                    continue
                if isinstance(reg_info_item, str):
                    reg_info_item = json.loads(reg_info_item)
                prefix = "_".join([prefix, reg_info_item.get(OP_NAME, "")])
            self.uniq_name = prefix + "_" + self.func_name
        else:
            raise TypeError("For '{}', 'func' must be of type function or str, but got {}"
                            .format(self.name, type(self.func)))

    def _update_reg_attrs(self, reg_info):
        """Update op attrs in reg_info."""
        output_name_list = []
        for _, item in enumerate(reg_info.get("outputs", [])):
            if isinstance(item, dict) and item.get(KEY_NAME):
                output_name_list.append(item.get(KEY_NAME))
        if output_name_list:
            self.add_prim_attr("output_names", output_name_list)

        if isinstance(reg_info.get(OP_NAME), str):
            self.add_prim_attr("reg_op_name", reg_info.get(OP_NAME))

        if self.func_type == "aicpu":
            self.uniq_name = reg_info[OP_NAME]
            self.add_prim_attr("uniq_name", self.uniq_name)

        if self.func_type in ["aot", "aicpu"]:
            if reg_info.get(KEY_ATTR) is not None and isinstance(reg_info[KEY_ATTR], list):
                for item in reg_info[KEY_ATTR]:
                    if isinstance(item, dict) and item.get("value") is not None:
                        self.add_prim_attr(item[KEY_NAME], item["value"])

    def _register_info(self, info):
        """Register reg_info."""
        reg_info = info
        if reg_info is None and hasattr(self.func, "reg_info"):
            reg_info = getattr(self.func, "reg_info")
        if self.func_type == "aicpu" and reg_info is None:
            raise ValueError("{}, the registration information must be set, but current is None. Use 'CustomRegOp' to "
                             "generate the registration information, then pass it to 'reg_info'"
                             .format(self.log_prefix))
        reg_info_list = self._get_expanded_list(reg_info)
        for reg_info in reg_info_list:
            if not isinstance(reg_info, (str, dict)):
                continue
            if isinstance(reg_info, str):
                reg_info = json.loads(reg_info)
            if self.fake_output:
                reg_info["outputs"].append(dict({"index": 0, KEY_NAME: "y", "param_type": "required"}))
                new_dtype_format = []
                for i in reg_info["dtype_format"]:
                    new_dtype_format.append(i + (DataType.I32_Default,))
                reg_info["dtype_format"] = new_dtype_format

            self._update_reg_attrs(reg_info)

            target = self._get_target(reg_info)
            # Reg info for func is only registered once for a certain target
            if self._has_registered(target):
                continue
            # Register
            reg_info = self._reformat_reg_info(reg_info, target)
            reg_info_str = json.dumps(reg_info)
            op_lib = Oplib()
            if not op_lib.reg_op(reg_info_str, self.imply_path):
                raise ValueError("{}, the registration information is registered failed. Use 'CustomRegOp' to "
                                 "generate the registration information, then pass it to 'reg_info' or use "
                                 "'custom_info_register' to bind it to 'func' if 'func' is a function."
                                 .format(self.log_prefix))
            self._save_attr(reg_info)
            self._save_register_status(target)

    def _get_expanded_list(self, data):
        """Recursive function to parse elements in list or tuple."""
        data_list = []
        if isinstance(data, (list, tuple)):
            for i in data:
                tmp_list = self._get_expanded_list(i)
                for ii in tmp_list:
                    data_list.append(ii)
        else:
            data_list.append(data)
        return data_list

    def _get_registered_targets(self):
        """Get the registered targets of func."""
        targets = []
        if callable(self.func):
            targets = getattr(self.func, "registered_targets", [])
        elif isinstance(self.func, str):
            targets = Custom.registered_func.get(self.uniq_name, [])
        if not isinstance(targets, list):
            targets = [targets]
        return targets

    def _has_registered(self, target):
        """Check if registration information is registered in target."""
        registered_targets = self._get_registered_targets()
        return target in registered_targets

    def _save_register_status(self, target):
        """Save registration status for target."""
        if callable(self.func):
            registered_targets = getattr(self.func, "registered_targets", [])
            registered_targets.append(target)
            setattr(self.func, "registered_targets", registered_targets)
        elif isinstance(self.func, str):
            func_name = self.uniq_name
            if isinstance(Custom.registered_func.get(func_name), list):
                Custom.registered_func.get(func_name).append(target)
            else:
                Custom.registered_func[func_name] = [target]

    def _reformat_reg_info(self, reg_info, target):
        """Reformat registration information."""
        if not isinstance(reg_info, dict):
            raise TypeError("{}, the registration information must be of type dict, but got {} with type {}. Use "
                            "'CustomRegOp' to generate the registration information, then pass it to 'reg_info' or "
                            "use 'custom_info_register' to bind it to 'func' if 'func' is a function."
                            .format(self.log_prefix, reg_info, type(reg_info)))
        reg_info[OP_NAME] = self.uniq_name
        reg_info[IMPLY_TYPE] = self._get_imply_type(reg_info, target)
        if not isinstance(reg_info.get(FUSION_TYPE), str) or not reg_info[FUSION_TYPE].strip():
            reg_info[FUSION_TYPE] = "OPAQUE"
        # Supplement necessary info for TBE if these information is missing in reg_info
        if reg_info[IMPLY_TYPE] == TBE:
            if reg_info.get(KEY_ATTR) is not None and isinstance(reg_info[KEY_ATTR], list):
                for i, item in enumerate(reg_info[KEY_ATTR]):
                    if isinstance(item, dict) and item.get("value") is None:
                        reg_info[KEY_ATTR][i]["value"] = "all"
            reg_info["async_flag"] = reg_info.get("async_flag", False)
            reg_info["binfile"] = "%s.so" % self.func_name
            reg_info["compute_cost"] = reg_info.get("compute_cost", 10)
            reg_info["kernel"] = self.func_name
            reg_info["partial_flag"] = reg_info.get("partial_flag", True)
            reg_info["needCheckSupport"] = reg_info.get("need_check_supported", False)
        # Supplement necessary info for AKG if these information is missing in reg_info
        if reg_info[IMPLY_TYPE] == AKG:
            target_to_processor = {ASCEND: AICORE, GPU: CUDA, CPU: CPU}
            reg_info["processor"] = reg_info.get("processor", target_to_processor.get(target))
        return reg_info

    def _get_target(self, reg_info):
        """Get target information."""
        target = None
        if isinstance(reg_info, dict):
            # Get target from reg_info["target"]
            target = reg_info.get("target")
            # Infer target from reg_info["processor"], reg_info generated from AkgGpuRegOp or AkgAscendRegOp
            #   will have the processor information.
            if target not in self.supported_targets:
                processor_to_target = {AICORE: ASCEND, CUDA: GPU, CPU: CPU}
                target = processor_to_target.get(reg_info.get("processor"))
            # Infer target from reg_info[IMPLY_TYPE]
            if target not in self.supported_targets:
                imply_type_to_target = {TBE: ASCEND, GPU: GPU, CPU: CPU}
                target = imply_type_to_target.get(reg_info.get(IMPLY_TYPE))
        # Infer target from func_type
        if target not in self.supported_targets:
            func_type_to_target = {"tbe": ASCEND, "pyfunc": CPU}
            target = func_type_to_target.get(self.func_type)
        if target not in self.supported_targets:
            raise ValueError("{}, target set in registration information must be one of {}, but got {}"
                             .format(self.log_prefix, self.supported_targets, target))
        return target

    def _get_imply_type(self, reg_info, target):
        """Get imply_typ information."""
        # Get imply_type from reg_info[IMPLY_TYPE]
        if isinstance(reg_info, dict) and isinstance(reg_info.get(IMPLY_TYPE), str) and \
                reg_info[IMPLY_TYPE].strip():
            return reg_info[IMPLY_TYPE]
        # Infer imply_type from func_type
        func_type_to_imply_type = {"hybrid": AKG, "akg": AKG, "tbe": TBE, "aicpu": "AiCPU", "pyfunc": target,
                                   "aot": "BiSheng" if target == ASCEND else target}
        return func_type_to_imply_type.get(self.func_type, AKG)

    def _save_attr(self, reg_info):
        """Save input_names and attr_names of current func."""
        if not isinstance(reg_info, dict):
            return

        def _get_value_list(key):
            value = reg_info.get(key, [])
            if not isinstance(value, (list, tuple)):
                value = [value]
            return value

        tensor_inputs = _get_value_list("inputs")
        attr = _get_value_list(KEY_ATTR)
        input_names = []  # include tensor input names and attr input names
        attr_names = []
        pure_input_names = []
        for item in tensor_inputs:
            if isinstance(item, dict) and item.get(KEY_NAME) is not None:
                input_names.append(item[KEY_NAME])
                pure_input_names.append(item[KEY_NAME])
        # attr is converted from inputs only when graph mode or when inputs name is also in reg info
        attr_to_input_safe = bool(input_names) or context.get_context("mode") == ms.GRAPH_MODE
        for item in attr:
            if isinstance(item, dict) and item.get(KEY_NAME) is not None:
                # for custom op with function tbe, we always add attrs to inputs as we don't
                # deal with attr value here and leave them to the backend process to fit the
                # usual process of tbe op compiling in mindspore
                # for the rest cases, namely aot and aicpu, if we find values for attrs, we
                # have already add them as prim attr of the op in the fun _update_reg_attrs
                # add attr name to input name only when the value of attr is None in reg info
                # as we need to get values of attrs from inputs
                if attr_to_input_safe and (self.func_type == "tbe" or item.get("value", None) is None):
                    input_names.append(item[KEY_NAME])
                attr_names.append(item[KEY_NAME])
        cur_attr = {INPUT_NAMES: input_names, ATTR_NAMES: attr_names, "pure_input_names": pure_input_names}
        # If func does not have attr, save current attr.
        # Else, check if current attr is same as previous saved one.
        prev_attr_names = attr_names
        if callable(self.func):
            func_attr = getattr(self.func, "func_attr", None)
            if not isinstance(func_attr, dict):
                setattr(self.func, "func_attr", cur_attr)
            else:
                prev_attr_names = func_attr.get(ATTR_NAMES)
        elif isinstance(self.func, str):
            func_attr = Custom.attr_dict.get(self.func)
            if not isinstance(func_attr, dict):
                Custom.attr_dict[self.func] = cur_attr
            else:
                prev_attr_names = func_attr.get(ATTR_NAMES)
        if attr_names != prev_attr_names:
            raise ValueError("{}, attr names set in registration information must be the same as previous saved one, "
                             "but got {} vs {}".format(self.log_prefix, attr_names, prev_attr_names))

    def _add_prim_target(self):
        """Add primitive_target to primitive's attr."""
        registered_targets = self._get_registered_targets()
        if self.func_type == "pyfunc":
            self.set_device(CPU)
            if registered_targets and registered_targets != [CPU]:
                logger.warning("{}, only supports CPU platform, but got registered target {}. "
                               "We will run it on CPU".format(self.log_prefix, registered_targets))
        elif self.func_type == "aot":
            if len(registered_targets) != 1:
                logger.info("{}, target will be set according to context.".format(self.log_prefix))
            elif registered_targets == [GPU]:
                self.set_device(GPU)
            elif registered_targets == [CPU]:
                self.set_device(CPU)

    def _update_attr(self):
        """Add input_names, attr_names, primitive_target to primitive's attr."""

        def _add_prim_attr(key):
            value = func_attr.get(key)
            if value:
                self.add_prim_attr(key, value)

        func_attr = {}
        if callable(self.func):
            inputs_num = len(inspect.signature(self.func).parameters)
            self.add_prim_attr("inputs_num", inputs_num)
            func_attr = getattr(self.func, "func_attr", None)
        elif isinstance(self.func, str):
            func_attr = Custom.attr_dict.get(self.func)
        if isinstance(func_attr, dict):
            _add_prim_attr(INPUT_NAMES)
            _add_prim_attr(ATTR_NAMES)
            _add_prim_attr("pure_input_names")
        self._add_prim_target()
        if callable(self.func) and callable(self.out_shape):
            if hasattr(self.out_shape, "type") and getattr(self.out_shape, "type") == AUTO_DIFF:
                self.add_prim_attr(AUTO_DIFF, True)
        else:
            self.add_prim_attr(AUTO_DIFF, False)

    def _hybrid_autodiff(self, input_func_type):
        """generate backward op for a custom hybrid op"""
        inputs_num = len(inspect.signature(self.func).parameters)
        if inputs_num == 0:
            logger.warning("{}, 'func' with no input has no backward operator.".format(self.log_prefix))
        elif inputs_num > 10:
            logger.warning("{}, currently automatic differentiation for 'func' with more than 10 inputs is not "
                           "supported.".format(self.log_prefix))
        else:
            grad_func = autodiff_bprop(inputs_num)

            def infer_func(*args):
                return args[:inputs_num]

            setattr(infer_func, "type", AUTO_DIFF)
            op = Custom(func=self.func, out_shape=infer_func, out_dtype=infer_func,
                        func_type=input_func_type, bprop=True)
            self.bprop = grad_func(op)

    def _hybrid_func_analyser(self):
        """analyze hybrid source string and add corresponding attrs."""
        args = {val: idx for idx, val in enumerate(list(inspect.signature(self.func).parameters))}
        return_num = self.func_source_str.count('return')
        if return_num != 1:
            logger.warning("{}, source code of 'func' should have only one 'return' syntax, but got {}"
                           .format(self.log_prefix, return_num))
        else:
            sentences = [s for s in self.func_source_str.split('\n') if s.count("return") == 1]
            symbols = re.sub(r"return|\s|\[|\]|\(|\)", "", sentences[-1]).split(',')
            inplace_assign_output = [[idx, args[val]] if val in args else [idx, -1]
                                     for idx, val in enumerate(symbols)]

            if any(i[1] != -1 for i in inplace_assign_output):
                self.add_prim_attr("inplace_assign_output", " ".join(
                    (str(j) for i in inplace_assign_output for j in i)))

    def _auto_infer(self, *args):
        """
        the automatic infer function for functions with @kernel decorator
        """
        fake_input = []
        enable_infer_value = True
        for arg in args:
            if arg["value"] is not None:
                fake_input.append(arg["value"].asnumpy())
            else:
                arg_dtype = arg["dtype"]
                # if any value is missing from input, disable infer value
                enable_infer_value = False
                if isinstance(arg_dtype, mstype.TensorType):
                    arg_dtype = arg_dtype.element_type()
                fake_arg = np.zeros(arg["shape"]).astype(
                    mstype._dtype_to_nptype(arg_dtype))  # pylint:disable=protected-access
                fake_input.append(fake_arg)

        fake_output = self.func(*fake_input)

        if hasattr(fake_output, 'shape'):
            infer_shape = fake_output.shape
            # pylint:disable=protected-access
            infer_dtype = mstype.TensorType(mstype._pytype_to_dtype(fake_output.dtype))
        else:
            infer_shape = (1,)
            infer_dtype = mstype._pytype_to_dtype(fake_output.dtype)  # pylint:disable=protected-access

        infer_value = Tensor(fake_output) if enable_infer_value else None

        return infer_shape, infer_dtype, infer_value

    def _generate_reg_info(self):
        if not self.is_ascend_c:
            return
        if self.reg_info is None:
            func_name, _ = self._split_func()
            if func_name.startswith("aclnn"):
                func_name = func_name[len("aclnn"):]
            reg_info_generator = CustomInfoGenerator(func_name)
            self.reg_info = reg_info_generator.generate_custom_reg_op()

    def _split_func(self):
        func_list = self.func.split(":")
        func_path = ""
        if len(func_list) == 2:
            func_path = func_list[0]
            func_name = func_list[1]
        else:
            func_name = self.func
        return func_name, func_path

    def _generate_get_worspace_size_func_by_types(self, aclnn_api_types):
        """generate custom GetWorkSpaceSize func by aclnn api types"""
        if not self.is_ascend_c:
            return

        input_output_types = []
        if isinstance(aclnn_api_types, str):
            params = re.split(r',\s*', aclnn_api_types)
            for param in params:
                param = param.replace('const ', '')
                type_part = re.search(r'^\s*(\w+\s*\*+|\w+)', param).group(1)
                type_part = type_part.replace(' ', '')
                input_output_types.append(type_part)
        elif isinstance(aclnn_api_types, list):
            input_output_types = aclnn_api_types
        else:
            raise RuntimeError(f"Unsupported type: {type(aclnn_api_types)}, support type is list or string.")

        func_name, _ = self._split_func()
        file_path = os.path.join(_get_cache_path(), func_name, func_name + "_callback.cc")

        file_path = os.path.abspath(file_path)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        custom_builder = CustomCodeGenerator()
        callback_func = custom_builder.generate_callback_by_types(func_name, self.reg_info, input_output_types)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(callback_func)

        custom_callback_func_path = _compile_aot(file_path)
        custom_callback_func = custom_callback_func_path + ":" + func_name
        self.add_prim_attr("custom_callback_func", custom_callback_func)
        self.add_prim_attr("custom_inputs_type", input_output_types[:-2])

    def _generate_get_workspace_size_func(self):
        """generate custom GetWorkSpaceSize func"""
        if not self.is_ascend_c:
            return
        func_name, _ = self._split_func()
        file_path = os.path.join(_get_cache_path(), func_name, func_name + "_callback.cc")

        file_path = os.path.abspath(file_path)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        custom_info_generator = CustomInfoGenerator(func_name)
        api_types = custom_info_generator.get_aclnn_api_types()
        custom_builder = CustomCodeGenerator()
        if not api_types:
            api_types = custom_builder.get_api_types_by_reg_info(self.reg_info)

        callback_func = custom_builder.generate_callback_by_types(func_name, self.reg_info, api_types)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(callback_func)

        custom_callback_func_path = _compile_aot(file_path)
        custom_callback_func = custom_callback_func_path + ":" + func_name
        self.add_prim_attr("custom_callback_func", custom_callback_func)
        self.add_prim_attr("custom_inputs_type", api_types[:-2])

    def __call__(self, *args):
        if self.is_ascend_c:
            res = pyboost_custom_ext(self.custom_pyboost, [args])
            return res if self.multi_output else res[0]
        should_elim, output = self.check_elim(*args)
        if should_elim:
            return output
        # pylint: disable=protected-access
        return ops.primitive._run_op(self, self.name, args)


class _MultiSoProxy:
    """
    A thin wrapper that transparently multiplexes attribute access between a
    pure-Python fallback module and an optional compiled shared-object (SO)
    module, honoring MindSpore’s current execution mode (GRAPH vs. PYNATIVE).
    """

    def __init__(self, func_module, so_module):
        """
        Args:
            func_module (module or None): Python module to serve as the fallback implementation source.
                May be ``None`` if no Python fallback is available.
            so_module (module): Compiled shared-object module that provides
                optimized kernels accessible only in ``PYNATIVE_MODE``.
        """
        self.func_module = func_module
        self.so_module = so_module

    def __getattr__(self, name: str):
        """
        Intercepts every attribute lookup and resolves it against the wrapped
        modules according to the documented precedence rules.

        Args:
            name (str): Name of the custom operation being requested.

        Returns:
            Any: The requested callable or attribute from either ``func_module`` or ``so_module``.

        Raises:
            AttributeError: If the attribute is not found in any applicable module or
            is incompatible with the current execution mode.
        """
        if self.func_module is not None and hasattr(self.func_module, name):
            return getattr(self.func_module, name)
        if context.get_context("mode") == ms.PYNATIVE_MODE:
            if hasattr(self.so_module, name):
                return getattr(self.so_module, name)
            raise AttributeError(
                f"Custom op '{name}' is neither in func_module nor in so_module."
            )

        raise AttributeError(
            f"Custom op '{name}' does not support GRAPH mode. "
            f"Please register it for GRAPH mode or switch to PYNATIVE mode."
        )


class CustomOpBuilder:
    r"""
    CustomOpBuilder is used to initialize and configure custom operators for MindSpore.
    Users can define and load custom operator modules through this class and apply them to the network.

    In most cases, users only need to provide the source files and additional compilation options in the constructor
    and call the `load` method to complete the compilation and loading of the operator.
    If users have specific customization requirements, they can inherit this class and override certain methods.
    It is important to note that if methods are overridden, some parameters passed to the constructor may be ignored.

    .. warning::
        This is an experimental API that is subject to change.

    Args:
        name (str): The unique name of the custom operator module, used to identify the operator.
        sources (Union[list[str], tuple[str], str]): The source file(s) of the custom operator. It can be a single
            file path or a list of file paths.
        backend (str, optional): The target backend for the operator, such as "CPU" or "Ascend". Default: ``None``.
        include_paths (Union[list[str], tuple[str], str], optional): Additionally included paths needed during
            compilation. Default: ``None``.
        cflags (str, optional): Extra C++ compiler flags to be used during compilation. Default: ``None``.
        ldflags (str, optional): Extra linker flags to be used during linking. Default: ``None``.
        kwargs (dict, optional): Additional keyword arguments for future extensions or specific custom requirements.

            - build_dir (str, optional): The directory used to generate the operator build files.
              If this argument is set, the provided path will be used directly.
              If not set, a subdirectory named after the operator's name will be created under the path specified by
              the environment variable `MS_COMPILER_CACHE_PATH` (defaulting to "./kernel_meta"), and the files will
              be placed in this subdirectory. Default: ``None``.

            - enable_atb (bool, optional): Whether to call ATB (Ascend Transformer Boost) operator. If set to ``True``,
              the `backend` must be ``Ascend`` or left empty. Default: ``False``.

            - enable_asdsip (bool, optional): Whether to call ASDSIP (Ascend SiP Boost) operator. If set to ``True``,
              the `backend` must be ``Ascend`` or left empty. Default: ``False``.

            - op_def (Union[list[str], tuple[str], str], optional): Path(s) to the operator definition
              file(s) (YAML format). When using custom operators in graph mode, this parameter is mandatory.
              It can be a single file path string or a list of file path strings. Default: ``None``.

            - op_doc (Union[list[str], tuple[str], str], optional): Path(s) to the operator documentation
              file(s) (YAML format). This parameter is optional and used to provide additional documentation
              for the operator. It can be a single file path string or a list of file path strings. Default: ``None``.

    .. note::
        - If the `backend` argument is provided, additional default flags will be automatically added to
          the compilation and linking steps to support the operator's target backend. The default options
          can be referenced in the implementation of the `get_cflags` and `get_ldflags` methods in the `CustomOpBuilder
          <https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/ops/operations/custom_ops.py>`_.
        - The `sources` argument must point to valid source files for the custom operator.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> builder = ops.CustomOpBuilder(
        ...     name="custom_op_cpu",
        ...     sources="custom_ops_impl/pybind_op_cpu.cpp",
        ...     backend="CPU"
        ... )
        >>> my_ops = builder.load()
    """
    _loaded_ops = {}

    def __init__(self, name, sources, backend=None, include_paths=None, cflags=None, ldflags=None, **kwargs):
        self._check_and_get_args(name, sources, backend, include_paths, cflags, ldflags, **kwargs)

        self._ms_path = os.path.dirname(os.path.abspath(ms.__file__))
        self.auto_generate = self.name + "_auto_generate"
        if self.enable_atb:
            if backend is not None and backend != "Ascend":
                raise ValueError("For 'CustomOpBuilder', when 'enable_atb' is set to True, the 'backend' must be "
                                 f"'Ascend' (or left implicit), but got '{backend}'")
            self.backend = "Ascend"
        if self.enable_asdsip:
            if backend is not None and backend != "Ascend":
                raise ValueError("For 'CustomOpBuilder', when 'enable_asdsip' is set to True, the 'backend' must be "
                                 f"'Ascend' (or left implicit), but got '{backend}'")
            self.backend = "Ascend"
        if self.backend == "Ascend":
            ascend_opp_path = os.getenv("ASCEND_OPP_PATH")
            if not ascend_opp_path:
                raise ValueError("Environment variable 'ASCEND_OPP_PATH' must be set for Ascend backend.")
            self.ascend_cann_path = ascend_opp_path.split('opp')[0]

            if self.enable_atb:
                self.atb_home_path = os.getenv("ATB_HOME_PATH")
                if not self.atb_home_path:
                    raise ValueError("Environment variable 'ATB_HOME_PATH' must be set when 'enable_atb' is True.")

    def _check_and_get_args(self, name, sources, backend=None, include_paths=None,
                            cflags=None, ldflags=None, **kwargs):
        """
        Validate and normalize all arguments to meet custom-op build requirements.
        """

        def _check_str_or_list_str(key, val):
            if val is None:
                return val
            if isinstance(val, str):
                val = [val]
            val = validator.check_value_type(key, val, [list, tuple])
            val = list(val)
            validator.check_element_type_of_iterable(key, val, [str])
            return val

        self.name = validator.check_value_type("name", name, [str])
        self.source = _check_str_or_list_str("sources", sources)
        self.backend = validator.check_value_type("backend", backend, [str, type(None)])
        if self.backend is not None and self.backend not in {"CPU", "Ascend"}:
            raise ValueError(
                f"For 'backend', only 'CPU' or 'Ascend' are allowed, but got '{self.backend}'.")

        self.include_paths = _check_str_or_list_str("include_paths", include_paths)

        self.cflags = validator.check_value_type("cflags", cflags, [str, type(None)])
        self.ldflags = validator.check_value_type("ldflags", ldflags, [str, type(None)])

        self.build_dir = validator.check_value_type("build_dir",
                                                    kwargs.get("build_dir"),
                                                    [str, type(None)])

        self.debug_mode = validator.check_bool(kwargs.get("debug_mode", False), "debug_mode")
        self.enable_asdsip = validator.check_bool(kwargs.get("enable_asdsip", False), "enable_asdsip")
        self.yaml = _check_str_or_list_str("op_def", kwargs.get("op_def"))
        self.doc = _check_str_or_list_str("op_doc", kwargs.get("op_doc"))

        self.enable_atb = validator.check_bool(kwargs.get("enable_atb", False))

    def _generate_custom_op_def(self, module: str, input_path: str, doc_path: str, output_path: str) -> None:
        """Call gen_custom_ops.py to generate custom operator definition"""
        file_path = os.path.join(self._ms_path, "ops_generate/gen_custom_ops.py")
        cmd = [
            sys.executable,
            file_path,
            "-i", input_path,
            "-o", output_path,
            "-m", module,
            "-d", doc_path
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"gen_custom_op.py failed with exit code {exc.returncode}.\n"
                f"stdout: {exc.stdout}\n"
                f"stderr: {exc.stderr}"
            ) from None

    def _get_op_def(self):
        """
        Generate C++ operator-definition source files from one or more YAML specification files.
        """
        if self.yaml is None:
            return []

        if self.doc is None:
            logger.info("Missing required 'doc': no YAML document was provided.")

        build_path = self._get_build_directory()
        yaml_path = os.path.join(build_path, "yaml")
        op_def_path = os.path.join(build_path, self.auto_generate)
        if os.path.exists(op_def_path):
            shutil.rmtree(op_def_path)
        os.makedirs(op_def_path, exist_ok=True)

        def copy_files(yaml_files, dest_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            os.makedirs(dest_path, exist_ok=True)
            for file_path in yaml_files:
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                filename = os.path.basename(file_path)
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in ('.yaml', '.yml'):
                    raise ValueError(f"Invalid file extension: {file_ext} for {filename}")

                _dest_path = os.path.join(dest_path, filename)
                shutil.copy2(file_path, _dest_path)

        yaml_files = [self.yaml] if isinstance(self.yaml, str) else self.yaml
        copy_files(yaml_files, yaml_path)
        doc_path = ""
        if self.doc is not None:
            doc_path = os.path.join(build_path, "doc")
            doc_files = [self.doc] if isinstance(self.doc, str) else self.doc
            copy_files(doc_files, doc_path)

        self._generate_custom_op_def(self.name, yaml_path, doc_path, op_def_path)
        return [os.path.join(op_def_path, "gen_custom_ops_def.cc")]

    def get_sources(self):
        """
        Get the source files for the custom operator.

        Returns:
            str or list[str], The source file(s) for the operator.
        """
        self.source = [self.source] if isinstance(self.source, str) else self.source
        return self.source + self._get_op_def()

    def get_include_paths(self):
        """
        Get the include paths required for compiling the custom operator.

        Returns:
            list[str], A list of include paths.
        """
        include_list = self.include_paths if self.include_paths is not None else []
        include_list.append(self._ms_path)
        include_list.append(os.path.join(self._ms_path, "include"))
        include_list.append(os.path.join(self._ms_path, "include", "third_party"))
        include_list.append(os.path.join(self._ms_path, "include", "third_party", "robin_hood_hashing"))
        include_list.append(os.path.join(self._ms_path, "include", "third_party", "securec", "include"))

        if self.backend == "Ascend":
            include_list.append(os.path.join(self.ascend_cann_path, "include"))
            include_list.append(os.path.join(self.ascend_cann_path, "include/external"))
            if self.enable_atb:
                include_list.append(os.path.join(self.atb_home_path, "include"))
        include_list += self._get_ms_inner_includes()
        return include_list

    def _get_ms_inner_includes(self):
        """include paths for inner module interface."""
        ms_inner_path = os.path.join(self._ms_path, "include", "mindspore")
        include_list = []
        include_list.append(os.path.join(ms_inner_path, "include"))
        include_list.append(os.path.join(ms_inner_path, "core", "include"))
        include_list.append(os.path.join(ms_inner_path, "core", "mindrt", "include"))
        include_list.append(os.path.join(ms_inner_path, "core", "mindrt"))
        include_list.append(os.path.join(ms_inner_path, "ops"))
        include_list.append(os.path.join(ms_inner_path, "ops", "include"))
        include_list.append(os.path.join(ms_inner_path, "ops", "kernel", "include"))
        include_list.append(os.path.join(ms_inner_path, "ccsrc"))
        include_list.append(os.path.join(ms_inner_path, "ccsrc", "include"))
        include_list.append(os.path.join(ms_inner_path, "ccsrc", "minddata", "mindrecord", "include"))
        return include_list

    def get_cflags(self):
        """
        Get the C++ compiler flags for building the custom operator.

        Returns:
            list[str], A list of C++ compiler flags.
        """
        flags = [f'-DMS_EXTENSION_NAME={self.name}', '-D_GLIBCXX_USE_CXX11_ABI=0', '-DENABLE_FAST_HASH_TABLE=1']
        flags += ['-std=c++17', '-fstack-protector-all', '-fPIC', '-pie']
        if self.debug_mode:
            flags.append('-g')
        else:
            flags.append('-O2')
        if self.backend == "Ascend":
            flags.append('-DCUSTOM_ASCEND_OP')
            if self.enable_atb:
                flags.append('-DCUSTOM_ENABLE_ATB')
            if self.enable_asdsip:
                flags.append('-DCUSTOM_ENABLE_ASDSIP')
        if self.cflags is not None:
            flags.append(self.cflags)
        return flags

    def get_ldflags(self):
        """
        Get the linker flags for building the custom operator.

        Returns:
            list[str], A list of linker flags.
        """
        flags = ['-shared']
        flags += ['-Wl,-z,relro,-z,now,-z,noexecstack', '-Wl,--disable-new-dtags,--rpath']
        if not self.debug_mode:
            flags.append('-s')  # strip
        flags += [
            f"-L{os.path.abspath(os.path.join(self._ms_path, 'lib'))}",
            '-lmindspore_core',
            '-lmindspore_ms_backend',
            '-lmindspore_pynative',
            '-lmindspore_pyboost'
        ]
        if self.backend == "Ascend":
            flags.append(f"-L{os.path.abspath(os.path.join(self.ascend_cann_path, 'lib64'))}")
            flags.append('-lascendcl')
            plugin_path = os.path.abspath(os.path.join(self._ms_path, 'lib', 'plugin'))
            flags.append(f"-L{plugin_path}")
            flags.append(f"-L{os.path.join(plugin_path, 'ascend')}")
            flags.append('-l:libmindspore_ascend.so.2')
            flags.append('-lmindspore_extension_ascend_aclnn')
            if self.enable_atb:
                flags.append('-lmindspore_extension_ascend_atb')
                flags.append(f"-L{os.path.abspath(os.path.join(self.atb_home_path, 'lib'))}")
                flags.append('-latb')
            if self.enable_asdsip:
                flags.append(f"-L{os.path.abspath(os.path.join(self._ms_path, 'lib', 'plugin', 'ascend'))}")
                flags.append('-lmindspore_extension_ascend_asdsip')
        if self.ldflags is not None:
            flags.append(self.ldflags)
        return flags

    def build(self):
        """
        Build the custom operator module.

        This method generates a dynamic library file for the custom operator based on the provided source files,
        include paths, compilation flags, and link flags.

        Returns:
            str, The path to the compiled module.
        """
        return ExtensionBuilder(self._get_build_directory()).build(
            module_name=self.name,
            sources=self.get_sources(),
            extra_include_paths=self.get_include_paths(),
            extra_cflags=self.get_cflags(),
            extra_ldflags=self.get_ldflags())

    def load(self):
        """
        Build and load the custom operator module.

        Returns:
            Module, The loaded custom operator module.
        """
        if self.name in CustomOpBuilder._loaded_ops:
            return CustomOpBuilder._loaded_ops[self.name]

        module_path = self.build()
        so_module = CustomOpBuilder._import_module(module_path)
        func_module = None
        if self.yaml is not None:
            module_path = os.path.join(self.build_dir, self.auto_generate, "gen_ops_def.py")
            sys.path.append(os.path.join(self.build_dir, self.auto_generate))
            sys.path.append(os.path.join(self.build_dir))
            func_module = self._import_module(module_path, True)
        mod = _MultiSoProxy(func_module, so_module)
        CustomOpBuilder._loaded_ops[self.name] = mod
        return mod

    @staticmethod
    def _import_module(module_path, is_yaml_build=False):
        """Import module from library."""
        module_path = os.path.abspath(module_path)
        module_dir = os.path.dirname(module_path)
        module_name = os.path.splitext(os.path.basename(module_path))[0]

        if is_yaml_build:
            package_name = os.path.basename(module_dir)
            if module_dir not in sys.path:
                sys.path.append(module_dir)

            if package_name not in sys.modules:
                pkg_spec = importlib.machinery.ModuleSpec(package_name, None, is_package=True)
                pkg = importlib.util.module_from_spec(pkg_spec)
                pkg.__path__ = [module_dir]
                sys.modules[package_name] = pkg

            module_name = f"{package_name}.{module_name}"

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _get_build_directory(self):
        """Get build directory."""
        if self.build_dir is None:
            build_root = os.path.realpath(os.getenv('MS_COMPILER_CACHE_PATH', "./kernel_meta"))
            self.build_dir = os.path.join(build_root, self.name)
        else:
            self.build_dir = os.path.realpath(self.build_dir)
        logger.info(f'Build {self.name} in directory {self.build_dir}')
        if not os.path.exists(self.build_dir):
            os.makedirs(self.build_dir, exist_ok=True)
        return self.build_dir


def _make_sigs(arg_specs):
    """
    Convert argument specifications into MindSpore operator signatures.
    """

    return tuple(
        sig.make_sig(name)
        for name, io, _ in arg_specs
        if io == "input"
    )


def _map_type_to_spec(typ: type) -> str:
    """
    Translate a Python type annotation to the string code expected by PyFunc.

    Raises
    ------
    TypeError
        If *typ* is not supported.
    """
    # 1. Tensor
    if typ is Tensor:
        return "tensor"

    # 2. Scalar
    _signature_type_map = {
        Tensor: "tensor",
        int: "int",
        float: "float",
        bool: "bool",
    }
    try:
        return _signature_type_map[typ]
    except KeyError:
        pass

    origin = getattr(typ, "__origin__", None)
    args = getattr(typ, "__args__", ())

    # 3. List[Element]
    if origin is list and args:
        elem = args[0]
        if elem is Tensor:
            return "list_tensor"
        if elem in (int, float, bool):
            return f"list_{elem.__name__}"

    # 4. Tuple[Element, ...]
    if origin is tuple and args:
        elem = args[0]
        if elem is Tensor:
            return "tuple_tensor"
        if elem in (int, float, bool):
            return f"tuple_{elem.__name__}"

    raise TypeError(f"Type annotation {typ!r} is not supported by @ms_pyfunc")


class _RegistryManager:
    """
    Thread-safe container that ensures each operator name is registered
    exactly once.  Internal use only.
    """

    def __init__(self) -> None:
        self._registry: set[str] = set()
        self._lock = threading.Lock()

    def register(self, name: str, builder) -> None:
        """
        Register the operator described by *builder* if *name* has not
        been processed before.  The check-and-register sequence is
        protected by a re-entrant lock to guarantee safety across
        concurrent decorator applications.
        """
        with self._lock:
            if name not in self._registry:
                builder.register_op()
                self._registry.add(name)


# Singleton instance used by the decorator factory
_registry_manager = _RegistryManager()


def _ms_pyfunc(infer_func: Callable[..., Any]) -> Callable[[Callable[..., Any]], Primitive]:
    """
    Decorator factory that converts a Python function into a MindSpore
    primitive operator.

    Args:
        infer_func (callable): A shape/dtype inference function that will be attached to the operator.

    Returns:
         A callable decorator that turns *pyfunc_fn* into a :class:`~mindspore.ops.Primitive`.
    """

    def decorator(pyfunc_fn: Callable[..., Any]) -> Primitive:
        # 1. Parse type annotations ------------------------------------------------
        fn_sig = inspect.signature(pyfunc_fn)
        hints = get_type_hints(pyfunc_fn)

        arg_specs: list[tuple[str, str, str]] = []
        for name, _ in fn_sig.parameters.items():
            spec = _map_type_to_spec(hints.get(name, type(None)))
            arg_specs.append((name, "input", spec))

        ret_spec = _map_type_to_spec(hints.get("return", type(None)))
        arg_specs.append(("out", "output", ret_spec))

        # 2. Register operator -----------------------------------------------------
        op_name = pyfunc_fn.__name__
        prim_cls_name = f"PyCallback_{op_name}"

        builder = PyFuncOpDefBuilder(prim_cls_name)
        for n, r, t in arg_specs:
            builder.arg(n, r, t)

        # Thread-safe registration through the private manager
        _registry_manager.register(op_name, builder)

        func_id = id(pyfunc_fn)
        infer_id = id(infer_func)
        add_pyfunc(func_id, pyfunc_fn)
        add_pyfunc(infer_id, infer_func)

        def __init__(self: Primitive) -> None:
            super(self.__class__, self).__init__(self.__class__.__name__)
            self.add_prim_attr("fn_id", func_id)
            self.add_prim_attr("infer_func_id", infer_id)
            self.add_prim_attr("prim_py_func", True)

        _prim_cls = type(
            prim_cls_name,
            (Primitive,),
            {
                "__mindspore_signature__": _make_sigs(tuple(arg_specs)),
                "__init__": prim_arg_register(__init__),
            },
        )
        _prim_singleton = _prim_cls()

        return _prim_singleton

    return decorator
