/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <memory>
#include "utils/ms_context.h"
#include "utils/log_adapter.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
namespace {
void MsCtxSetParameter(const std::shared_ptr<MsContext> &ctx, MsCtxParam param, const py::object &value) {
  MS_LOG(DEBUG) << "set param(" << param << ") with value '" << py::str(value).cast<std::string>() << "' of type '"
                << py::str(value.get_type()).cast<std::string>() << "'.";
  if (param >= MS_CTX_TYPE_BOOL_BEGIN && param < MS_CTX_TYPE_BOOL_END && py::isinstance<py::bool_>(value)) {
    ctx->set_param<bool>(param, value.cast<bool>());
    return;
  }
  if (param >= MS_CTX_TYPE_INT_BEGIN && param < MS_CTX_TYPE_INT_END && py::isinstance<py::int_>(value)) {
    ctx->set_param<int>(param, value.cast<int>());
    return;
  }
  if (param >= MS_CTX_TYPE_UINT32_BEGIN && param < MS_CTX_TYPE_UINT32_END && py::isinstance<py::int_>(value)) {
    ctx->set_param<uint32_t>(param, value.cast<uint32_t>());
    return;
  }
  if (param >= MS_CTX_TYPE_FLOAT_BEGIN && param < MS_CTX_TYPE_FLOAT_END && py::isinstance<py::float_>(value)) {
    ctx->set_param<float>(param, value.cast<float>());
    return;
  }
  if (param >= MS_CTX_TYPE_STRING_BEGIN && param < MS_CTX_TYPE_STRING_END && py::isinstance<py::str>(value)) {
    ctx->set_param<std::string>(param, value.cast<std::string>());
    return;
  }

  MS_LOG(EXCEPTION) << "Got illegal param " << param << " and value with type "
                    << py::str(value.get_type()).cast<std::string>();
}

py::object MsCtxGetParameter(const std::shared_ptr<MsContext> &ctx, MsCtxParam param) {
  if (param >= MS_CTX_TYPE_BOOL_BEGIN && param < MS_CTX_TYPE_BOOL_END) {
    return py::bool_(ctx->get_param<bool>(param));
  }
  if (param >= MS_CTX_TYPE_INT_BEGIN && param < MS_CTX_TYPE_INT_END) {
    return py::int_(ctx->get_param<int>(param));
  }
  if (param >= MS_CTX_TYPE_UINT32_BEGIN && param < MS_CTX_TYPE_UINT32_END) {
    return py::int_(ctx->get_param<uint32_t>(param));
  }
  if (param >= MS_CTX_TYPE_FLOAT_BEGIN && param < MS_CTX_TYPE_FLOAT_END) {
    return py::float_(ctx->get_param<float>(param));
  }
  if (param >= MS_CTX_TYPE_STRING_BEGIN && param < MS_CTX_TYPE_STRING_END) {
    return py::str(ctx->get_param<std::string>(param));
  }

  MS_LOG(EXCEPTION) << "Got illegal param " << param << ".";
}
}  // namespace

// Note: exported python enum variables beginning with '_' are for internal use
void RegMsContext(const py::module *m) {
  (void)py::enum_<MsCtxParam>(*m, "ms_ctx_param", py::arithmetic())
    .value("check_bprop", MsCtxParam::MS_CTX_CHECK_BPROP_FLAG)
    .value("enable_dump", MsCtxParam::MS_CTX_ENABLE_DUMP)
    .value("enable_graph_kernel", MsCtxParam::MS_CTX_ENABLE_GRAPH_KERNEL)
    .value("enable_reduce_precision", MsCtxParam::MS_CTX_ENABLE_REDUCE_PRECISION)
    .value("precompile_only", MsCtxParam::MS_CTX_PRECOMPILE_ONLY)
    .value("enable_profiling", MsCtxParam::MS_CTX_ENABLE_PROFILING)
    .value("save_graphs", MsCtxParam::MS_CTX_SAVE_GRAPHS_FLAG)
    .value("enable_parallel_split", MsCtxParam::MS_CTX_ENABLE_PARALLEL_SPLIT)
    .value("max_device_memory", MsCtxParam::MS_CTX_MAX_DEVICE_MEMORY)
    .value("mempool_block_size", MsCtxParam::MS_CTX_MEMPOOL_BLOCK_SIZE)
    .value("mode", MsCtxParam::MS_CTX_EXECUTION_MODE)
    .value("device_target", MsCtxParam::MS_CTX_DEVICE_TARGET)
    .value("inter_op_parallel_num", MsCtxParam::MS_CTX_INTER_OP_PARALLEL_NUM)
    .value("runtime_num_threads", MsCtxParam::MS_CTX_RUNTIME_NUM_THREADS)
    .value("_graph_memory_max_size", MsCtxParam::MS_CTX_GRAPH_MEMORY_MAX_SIZE)
    .value("print_file_path", MsCtxParam::MS_CTX_PRINT_FILE_PATH)
    .value("profiling_options", MsCtxParam::MS_CTX_PROFILING_OPTIONS)
    .value("save_dump_path", MsCtxParam::MS_CTX_SAVE_DUMP_PATH)
    .value("deterministic", MsCtxParam::MS_CTX_DETERMINISTIC)
    .value("save_graphs_path", MsCtxParam::MS_CTX_SAVE_GRAPHS_PATH)
    .value("compile_cache_path", MsCtxParam::MS_CTX_COMPILE_CACHE_PATH)
    .value("variable_memory_max_size", MsCtxParam::MS_CTX_VARIABLE_MEMORY_MAX_SIZE)
    .value("device_id", MsCtxParam::MS_CTX_DEVICE_ID)
    .value("auto_tune_mode", MsCtxParam::MS_CTX_TUNE_MODE)
    .value("max_call_depth", MsCtxParam::MS_CTX_MAX_CALL_DEPTH)
    .value("env_config_path", MsCtxParam::MS_CTX_ENV_CONFIG_PATH)
    .value("graph_kernel_flags", MsCtxParam::MS_CTX_GRAPH_KERNEL_FLAGS)
    .value("grad_for_scalar", MsCtxParam::MS_CTX_GRAD_FOR_SCALAR)
    .value("pynative_synchronize", MsCtxParam::MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE)
    .value("disable_format_transform", MsCtxParam::MS_CTX_DISABLE_FORMAT_TRANSFORM)
    .value("memory_offload", MsCtxParam::MS_CTX_ENABLE_MEM_OFFLOAD)
    .value("memory_optimize_level", MsCtxParam::MS_CTX_MEMORY_OPTIMIZE_LEVEL)
    .value("op_timeout", MsCtxParam::MS_CTX_OP_TIMEOUT);
  (void)py::class_<mindspore::MsContext, std::shared_ptr<mindspore::MsContext>>(*m, "MSContext")
    .def_static("get_instance", &mindspore::MsContext::GetInstance, "Get ms context instance.")
    .def("get_param", &mindspore::MsCtxGetParameter, "Get value of specified parameter.")
    .def("set_param", &mindspore::MsCtxSetParameter, "Set value for specified parameter.")
    .def("get_backend_policy", &mindspore::MsContext::backend_policy, "Get backend policy.")
    .def("set_backend_policy", &mindspore::MsContext::set_backend_policy, "Set backend policy.")
    .def("enable_dump_ir", &mindspore::MsContext::enable_dump_ir, "Get the ENABLE_DUMP_IR.")
    .def("is_ascend_plugin_loaded", &mindspore::MsContext::IsAscendPluginLoaded,
         "Get the status that has ascend plugin been loaded.")
    .def("register_set_env_callback", &mindspore::MsContext::RegisterSetEnv,
         "Register callback function for check environment variable.")
    .def("register_check_env_callback", &mindspore::MsContext::RegisterCheckEnv,
         "Register callback function for check environment variable.");
}
}  // namespace mindspore
