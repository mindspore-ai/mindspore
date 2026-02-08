/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/executor/ge_utils.h"

#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <nlohmann/json.hpp>
#include "include/utils/anfalgo.h"
#include "backend/ge_backend/graph_ir/types.h"
#include "backend/ge_backend/graph_ir/utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "utils/phase.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "plugin/ascend/res_manager/device_context_conf/op_tuning_conf.h"
#include "plugin/ascend/res_manager/device_context_conf/op_precision_conf.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
namespace {
constexpr char kGeDumpMode[3][7] = {"all", "input", "output"};
constexpr uint32_t kDumpModeAicoreOverflow = 1;
constexpr uint32_t kDumpModeAtomicOverflow = 2;
constexpr uint32_t kDumpModeAll = 3;

std::string ShapesToString(const ShapeArray &shapes) {
  std::stringstream buffer;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i != 0) {
      buffer << ",";
    }
    buffer << "[";
    const auto &shape = shapes[i];
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j != 0) {
        buffer << ",";
      }
      buffer << shape[j];
    }
    buffer << "]";
  }
  return buffer.str();
}

template <typename Map, typename K = typename Map::key_type, typename V = typename Map::mapped_type>
std::string MapToString(const Map &value) {
  std::stringstream buffer;
  buffer << "{";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it != value.begin()) {
      buffer << ", ";
    }
    buffer << it->first << ": " << it->second;
  }
  buffer << "}";
  return buffer.str();
}

inline std::string GetPhasePrefix() {
  const std::string &phase = PhaseManager::GetInstance().phase();
  auto pos = phase.find('.');
  if (pos != std::string::npos) {
    return phase.substr(0, pos);
  }

  return "";
}

void UpdateTopoOrderOptions(const string &graph_name, OptionMap *option) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);

  const auto &topo_order = context->get_param<std::string>(MS_CTX_TOPO_ORDER);
  if (topo_order.empty()) {
    return;
  }

  nlohmann::json topo_order_json = nlohmann::json::parse(topo_order);
  auto topo_order_iter = topo_order_json.find(graph_name);
  if (topo_order_iter == topo_order_json.end()) {
    return;
  }
  MS_LOG(INFO) << "Update topo order for graph " << graph_name << " to " << topo_order_iter.value();
  std::string topo_sorting_mode = "1";
  if (topo_order_iter.value() == "bfs") {
    topo_sorting_mode = "0";
  } else if (topo_order_iter.value() == "dfs") {
    topo_sorting_mode = "1";
  } else if (topo_order_iter.value() == "rdfs") {
    topo_sorting_mode = "2";
  }
  (*option)["ge.topoSortingMode"] = topo_sorting_mode;
}

void UpdatePrecisionOptions(const string &graph_name, OptionMap *option, bool is_cloud) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto soc_version = context->ascend_soc_version();

  auto op_precision_conf = device::ascend::OpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);
  auto precision_mode = op_precision_conf->precision_mode();
  if (!precision_mode.empty()) {
    MS_LOG(INFO) << "Set precision_mode " << precision_mode << " for graph " << graph_name << " by user.";
    (*option)["ge.exec.precision_mode"] = precision_mode;
  } else if (is_cloud && !IsTwoPhaseInfer()) {
    if (soc_version == "ascend910b" || soc_version == "ascend910_93") {
      (*option)["ge.exec.precision_mode"] = "must_keep_origin_dtype";
      MS_LOG(INFO) << "Set precision_mode must_keep_origin_dtype, soc_version is " << soc_version << ".";
    } else {
      (*option)["ge.exec.precision_mode"] = "allow_fp32_to_fp16";
      MS_LOG(INFO) << "Set precision_mode allow_fp32_to_fp16, soc_version is " << soc_version << ".";
    }
  } else {
    (*option)["ge.exec.precision_mode"] = "force_fp16";
    MS_LOG(INFO) << "Set precision_mode force_fp16, soc_version is " << soc_version << ".";
  }
}

OptionMap GetComputeGraphOptions(const ShapeArray &input_shapes, bool is_dynamic_shape) {
  OptionMap options{};
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto max_threshold = ms_context->get_param<std::string>(MS_CTX_HOST_SCHEDULING_MAX_THRESHOLD);
  if (!max_threshold.empty()) {
    (void)options.emplace("ge.exec.hostSchedulingMaxThreshold", max_threshold);
  }
  if (!is_dynamic_shape) {
    return options;
  }
  (void)options.emplace("ge.exec.dynamicGraphExecuteMode", "dynamic_execute");
  (void)options.emplace("ge.exec.dataInputsShapeRange", ShapesToString(input_shapes));
  return options;
}

void GetComputeGraphReuseOptions(const FuncGraphPtr &graph, OptionMap *option) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(option);
  auto enable_io_reuse = common::GetEnv("MS_ENABLE_IO_REUSE");
  MS_LOG(INFO) << "Enable io reuse: " << enable_io_reuse;
  if (enable_io_reuse != "1") {
    return;
  }
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  if (!outputs.empty()) {
    std::string value;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto output = outputs[i];
      const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
      auto &output_node = output_with_index.first;
      MS_EXCEPTION_IF_NULL(output_node);
      // Parameter and value can not been reused.
      if (output_node->isa<Parameter>() || output_node->isa<ValueNode>()) {
        MS_LOG(INFO) << "Output is parameter or value node, not support reuse, index is: " << i;
        continue;
      }
      (void)value.append(std::to_string(i));
      (void)value.append(",");
    }
    if (!value.empty()) {
      value.pop_back();
      MS_LOG(INFO) << "key: ge.exec.outputReuseMemIndexes, value: " << value << ",Graph name: " << graph->ToString();
      (void)option->insert(std::make_pair("ge.exec.outputReuseMemIndexes", value));
    }
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (graph->has_flag(backend::ge_backend::kGraphFlagHasGetNext) &&
      !graph->has_flag(backend::ge_backend::kGraphNeedIteration)) {
    MS_LOG(INFO) << "key: ge.exec.inputReuseMemIndexes, value: 0."
                 << ", Graph name: " << graph->ToString();
    (void)option->insert(std::make_pair("ge.exec.inputReuseMemIndexes", "0"));
  }
}
}  // namespace

bool IsGeTrain() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_ge = context->backend_policy() == "ge";
  bool enable_training = GetPhasePrefix() == "train";
  if (enable_ge && enable_training) {
    return true;
  }
  return false;
}

std::string GetGraphName(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  return graph->ToString();
}

void SetPassthroughGeOptions(std::string option_level, OptionMap *options) {
  const auto &new_options = AnfAlgo::GetGeOptions(option_level);
  for (auto &[key, value] : new_options) {
    (*options)[key] = value;
    MS_LOG(INFO) << "Set ge " << option_level << " option: {" << key << ", " << value << "}";
  }
}

bool AddFakeGraph(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = backend::ge_backend::NewConverter(anf_graph, GetPhasePrefix());
  backend::ge_backend::GenFakeGraph(anf_graph->ToString(), converter);
  auto graph_name = GetGraphName(anf_graph);
  std::string init_graph = "init_subgraph." + graph_name;
  ShapeArray shape_array;
  bool dynamic_shape_inputs = false;
  auto options = GetComputeGraphOptions(shape_array, dynamic_shape_inputs);
  GetComputeGraphReuseOptions(anf_graph, &options);
  UpdateTopoOrderOptions(graph_name, &options);
  MS_LOG(INFO) << "Set options of compute graph: " << graph_name << " to " << MapToString(options);
  (void)backend::ge_backend::AddGraph(graph_name, backend::ge_backend::GetComputeGraph(converter));
  (void)backend::ge_backend::AddGraph(init_graph, backend::ge_backend::GetInitGraph(converter));
  (void)backend::ge_backend::AddGraph(BROADCAST_GRAPH_NAME, backend::ge_backend::GetBroadcastGraph(converter));

  return true;
}

bool AddDFGraph(const FuncGraphPtr &anf_graph, const backend::ge_backend::TensorOrderMap &init_inputs_map,
                bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = backend::ge_backend::NewConverter(anf_graph, GetPhasePrefix());
  bool is_cloud = true;
  bool need_aoe = false;
  if (export_air) {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    backend::ge_backend::SetTraining(converter, false);
    backend::ge_backend::SetExportAir(converter, true);
    is_cloud = false;
  }
  backend::ge_backend::BuildGraph(anf_graph->ToString(), converter, init_inputs_map);
  backend::ge_backend::GenerateBroadcastGraph(converter, init_inputs_map);
  backend::ge_backend::GenerateCheckpointGraph(converter);
  auto err_code = backend::ge_backend::ErrCode(converter);
  if (err_code != 0) {
    backend::ge_backend::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return false;
  }

  if (device::ascend::OpTuningConf::GetInstance()->EnableAoeOnline()) {
    need_aoe = true;
  }
  auto graph_name = GetGraphName(anf_graph);
  std::string init_graph = "init_subgraph." + graph_name;
  auto options = GetComputeGraphOptions(converter->input_shapes(), converter->dynamic_shape_inputs());
  GetComputeGraphReuseOptions(anf_graph, &options);
  UpdateTopoOrderOptions(graph_name, &options);
  UpdatePrecisionOptions(graph_name, &options, is_cloud);

  MS_LOG(INFO) << "Set options of compute graph: " << graph_name << " to " << MapToString(options);
  (void)backend::ge_backend::AddGraph(graph_name, backend::ge_backend::GetComputeGraph(converter),
                                      backend::ge_backend::DfGraphConfig(options, is_cloud, need_aoe, export_air));
  (void)backend::ge_backend::AddGraph(init_graph, converter->GetInitGraph());
  (void)backend::ge_backend::AddGraph(BROADCAST_GRAPH_NAME, backend::ge_backend::GetBroadcastGraph(converter));
  return true;
}

void GetGeSessionOptions(backend::ge_backend::SessionOptions *options) {
  MS_EXCEPTION_IF_NULL(options);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  (*options)["ge.enablePrintOpPass"] = "0";
  (*options)["ge.constLifecycle"] = "graph";
  (*options)["ge.exec.formatMode"] = "0";
  auto format_mode = common::GetEnv("MS_FORMAT_MODE");
  if (format_mode == "1" || (format_mode.empty() && ms_context->ascend_soc_version() != "ascend910")) {
    MS_LOG(INFO) << "Set GE option ge.exec.formatMode to 1.";
    (*options)["ge.exec.formatMode"] = "1";
  }

  // options from set_context
  if (ms_context->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.graphMemoryMaxSize"] = ms_context->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE);
  }

  if (ms_context->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.variableMemoryMaxSize"] = ms_context->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
  }

  auto atomic_clean_policy = ms_context->get_param<std::string>(MS_CTX_ATOMIC_CLEAN_POLICY);
  if (atomic_clean_policy.empty()) {
    atomic_clean_policy = "1";
  }
  (*options)["ge.exec.atomicCleanPolicy"] = atomic_clean_policy;
  MS_LOG(INFO) << "Set GE atomic clean policy to " << atomic_clean_policy << ".";
  (*options)["ge.graphRunMode"] = "1";
}

void SavePrevStepWeight(const std::vector<AnfNodePtr> &weights, aclrtStream stream) {
  for (const auto &node : weights) {
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto out_kernel_tensor = AnfAlgo::GetOutputKernelTensor(param, 0, false);
      MS_EXCEPTION_IF_NULL(out_kernel_tensor);
      auto out_addr = out_kernel_tensor->device_address();
      if (out_addr == nullptr || out_addr->GetPtr() == nullptr ||
          IsOneOfHWSpecialFormat(kernel::GetFormatFromEnumToStr(out_kernel_tensor->format()))) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        continue;
      }
      auto size = tensor->Size();
      auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, tensor->data_c(), size, out_addr->GetMutablePtr(), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_SUCCESS) {
        MS_LOG_WITH_NODE(EXCEPTION, param) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
      tensor->set_copy_done_flag(true);
    }
  }
}
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
