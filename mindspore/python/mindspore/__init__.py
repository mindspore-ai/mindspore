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
""".. MindSpore package."""
from __future__ import absolute_import

from mindspore.run_check import run_check
from mindspore import common, dataset, mindrecord, train, log, amp, device_manager
from mindspore import profiler, communication, numpy, parallel, hal, runtime, device_context
from mindspore.common import *
from mindspore.common import _tensor_docs, bool, int, float
del _tensor_docs
from mindspore.mindrecord import *
from mindspore.ops import _op_impl, grad, value_and_grad, vjp, jvp, jacfwd, jacrev, vmap, get_grad, constexpr
from mindspore.train import *
from mindspore.log import *
from mindspore.utils import *
from mindspore.device_manager import *
from mindspore.runtime import *
from mindspore.context import GRAPH_MODE, PYNATIVE_MODE, set_context, get_context, set_auto_parallel_context, \
    get_auto_parallel_context, reset_auto_parallel_context, ParallelMode, set_ps_context, \
    get_ps_context, reset_ps_context, STRICT, COMPATIBLE, LAX
from mindspore.version import __version__
from mindspore.profiler import Profiler, EnvProfiler
from mindspore.parallel import set_algo_parameters, get_algo_parameters, reset_algo_parameters, \
    rank_list_for_transform, transform_checkpoint_by_rank, transform_checkpoints, merge_pipeline_strategys, shard, \
    Layout, sync_pipeline_shared_parameters, parameter_broadcast, load_segmented_checkpoints, unified_safetensors, \
    load_distributed_checkpoint, merge_sliced_parameter, restore_group_info_list, \
    build_searched_strategy
from mindspore.parallel.function import reshard
from mindspore.rewrite import SymbolTree, ScopedValue, Node, NodeType
from mindspore.safeguard import obfuscate_ckpt, load_obf_params_into_net
from mindspore.graph._utils.check_jit_forbidden_api import get_obj_module_and_name_info, is_jit_forbidden_module, \
    is_invalid_or_jit_forbidden_method
from mindspore import mint
from mindspore.ops._utils import arg_handler, arg_dtype_cast
from mindspore import onnx
from mindspore import graph
from mindspore import multiprocessing
from mindspore._c_expression import frombuffer

# Import the following modules to maintain compatibility
import mindspore.graph._utils.check_jit_forbidden_api as _check_jit_forbidden_api

__all__ = ["run_check"]
__all__.extend(__version__)
__all__.extend(common.__all__)
__all__.extend(train.__all__)
__all__.extend(log.__all__)
__all__.extend(context.__all__)
__all__.extend(parallel.__all__)
__all__.extend(rewrite.__all__)
__all__.extend(safeguard.__all__)
__all__.extend(device_manager.__all__)
__all__.extend(runtime.__all__)
__all__.extend(graph.__all__)
__all__.append("Profiler")
__all__.append("frombuffer")
