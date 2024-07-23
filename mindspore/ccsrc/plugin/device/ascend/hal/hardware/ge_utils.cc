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

#include "plugin/device/ascend/hal/hardware/ge_utils.h"

#include <tuple>
#include <utility>
#include <nlohmann/json.hpp>
#include "include/common/utils/anfalgo.h"
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/scoped_long_running.h"
#include "abstract/abstract_value.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "transform/symbol/symbol_utils.h"
#include "transform/symbol/acl_rt_symbol.h"
namespace mindspore {
namespace device {
namespace ascend {
using mindspore::transform::OptionMap;

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
  if (IsEnableRefMode()) {
    return graph->ToString();
  } else {
    KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
    std::string name;
    if (kg == nullptr) {
      name = graph->ToString();
    } else {
      FuncGraphPtr origin_graph = kg->GetFuncGraph();
      MS_EXCEPTION_IF_NULL(origin_graph);
      name = origin_graph->ToString();
    }
    return name;
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
  if (enable_io_reuse != "1" || !IsEnableRefMode()) {
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
  if (graph->has_flag(transform::kGraphFlagHasGetNext) && !graph->has_flag(transform::kGraphNeedIteration)) {
    MS_LOG(INFO) << "key: ge.exec.inputReuseMemIndexes, value: 0."
                 << ", Graph name: " << graph->ToString();
    (void)option->insert(std::make_pair("ge.exec.inputReuseMemIndexes", "0"));
  }
}

void SetPassthroughGeOptions(bool is_global, OptionMap *options) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);

  const auto &ge_options_str = context->get_param<std::string>(MS_CTX_GE_OPTIONS);
  if (ge_options_str.empty()) {
    MS_LOG(DEBUG) << "The ge option for passthrough is not set.";
    return;
  }

  string level = is_global ? "global" : "session";
  nlohmann::json options_json = nlohmann::json::parse(ge_options_str);
  auto options_iter = options_json.find(level);
  if (options_iter == options_json.end()) {
    MS_LOG(INFO) << "GE " << level << " option is not set.";
    return;
  }

  const auto &new_options = *options_iter;
  for (auto &[key, value] : new_options.items()) {
    (*options)[key] = value;
    MS_LOG(INFO) << "Set ge " << level << " option: {" << key << ", " << value << "}";
  }
}

namespace {
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
}  // namespace

bool AddFakeGraph(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = transform::NewConverter(anf_graph, GetPhasePrefix());
  transform::GenFakeGraph(anf_graph->ToString(), converter);
  auto graph_name = GetGraphName(anf_graph);
  std::string init_graph = "init_subgraph." + graph_name;
  std::string checkpoint_name = "save." + graph_name;
  ShapeArray shape_array;
  bool dynamic_shape_inputs = false;
  auto options = GetComputeGraphOptions(shape_array, dynamic_shape_inputs);
  GetComputeGraphReuseOptions(anf_graph, &options);
  UpdateTopoOrderOptions(graph_name, &options);
  MS_LOG(INFO) << "Set options of compute graph: " << graph_name << " to " << MapToString(options);
  (void)transform::AddGraph(graph_name, transform::GetComputeGraph(converter));
  (void)transform::AddGraph(init_graph, transform::GetInitGraph(converter));
  (void)transform::AddGraph(BROADCAST_GRAPH_NAME, transform::GetBroadcastGraph(converter));

  if (!IsEnableRefMode()) {
    transform::Status ret = transform::AddGraph(checkpoint_name, transform::GetSaveCheckpointGraph(converter));
    if (ret == transform::Status::SUCCESS) {
      transform::SetAnfGraph(checkpoint_name, anf_graph);
    }
  }
  return true;
}

bool AddDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map, bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = transform::NewConverter(anf_graph, GetPhasePrefix());
  bool is_cloud = true;
  bool need_aoe = false;
  if (export_air) {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    transform::SetTraining(converter, false);
    transform::SetExportAir(converter, true);
    is_cloud = false;
  }
  transform::BuildGraph(anf_graph->ToString(), converter, init_inputs_map);
  transform::GenerateBroadcastGraph(converter, init_inputs_map);
  transform::GenerateCheckpointGraph(converter);
  auto err_code = transform::ErrCode(converter);
  if (err_code != 0) {
    transform::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return false;
  }
  if (MsContext::GetInstance()->EnableAoeOnline()) {
    need_aoe = true;
  }
  auto graph_name = GetGraphName(anf_graph);
  std::string init_graph = "init_subgraph." + graph_name;
  std::string checkpoint_name = "save." + graph_name;
  auto options = GetComputeGraphOptions(converter->input_shapes(), converter->dynamic_shape_inputs());
  GetComputeGraphReuseOptions(anf_graph, &options);
  UpdateTopoOrderOptions(graph_name, &options);
  MS_LOG(INFO) << "Set options of compute graph: " << graph_name << " to " << MapToString(options);
  (void)transform::AddGraph(graph_name, transform::GetComputeGraph(converter), options, is_cloud, need_aoe);
  if (IsEnableRefMode()) {
    (void)transform::AddGraph(init_graph, converter->GetInitGraph());
  } else {
    (void)transform::AddGraph(init_graph, transform::GetInitGraph(converter));
  }
  (void)transform::AddGraph(BROADCAST_GRAPH_NAME, transform::GetBroadcastGraph(converter));

  if (!IsEnableRefMode()) {
    transform::Status ret = transform::AddGraph(checkpoint_name, transform::GetSaveCheckpointGraph(converter));
    if (ret == transform::Status::SUCCESS) {
      transform::SetAnfGraph(checkpoint_name, anf_graph);
    }
  }

  return true;
}

void SyncCopyStream(aclrtStream stream) {
  MS_LOG(INFO) << "Start sync copy data stream";
  if (CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream, -1) != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Exec aclrtSynchronizeStreamWithTimeout failed";
  }
  MS_LOG(INFO) << "End sync copy data stream";
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
      auto out_addr = AnfAlgo::GetMutableOutputAddr(param, 0, false);
      if (out_addr == nullptr || out_addr->GetPtr() == nullptr || IsOneOfHWSpecialFormat(out_addr->format())) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        continue;
      }
      auto size = tensor->Size();
      auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, tensor->data_c(), size, out_addr->GetMutablePtr(), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(EXCEPTION) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
      tensor->set_copy_done_flag(true);
    }
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
