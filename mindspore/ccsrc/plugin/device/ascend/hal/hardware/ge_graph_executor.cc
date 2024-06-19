/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ge_graph_executor.h"
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <map>
#include <set>
#include <sstream>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/draw.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/scoped_long_running.h"
#include "abstract/abstract_value.h"
#include "include/backend/kernel_graph.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/device/device_address_utils.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/optimizer/ge_optimization.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include "plugin/device/ascend/hal/device/ascend_device_synchronizer.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "ge/ge_graph_compile_summary.h"
#include "kernel/kernel_build_info.h"
#include "op_proto/inc/array_ops.h"
#include "ops/nn_op_name.h"
#include "ops/array_ops.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/compile_cache_context.h"
using InputNameAndType = std::vector<std::pair<std::string, bool>>;
using Data = ::ge::op::Data;
using RefData = ::ge::op::RefData;

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const std::set<std::string> kIgnoreGEShapeOps = {kSoftMarginLossOpName};
mindspore::HashMap<std::string, size_t> feature_memorys;
mindspore::HashMap<std::string, size_t> streams;
constexpr size_t kNeedRecycleOutput = 5;
constexpr int kCollectHostInfoStart = 0;
constexpr int kCollectHostInfoEnd = 1;

void GetMeRetDataType(const AbstractBasePtr &cnode_data, std::vector<TypeId> *me_types) {
  MS_EXCEPTION_IF_NULL(cnode_data);

  if (cnode_data->isa<abstract::AbstractNone>()) {
    return;
  }

  if (cnode_data->isa<abstract::AbstractTensor>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    if (me_type == kObjectTypeTensorType) {
      me_type = dyn_cast<TensorType>(cnode_data->BuildType())->element()->type_id();
      (void)me_types->emplace_back(me_type);
    }
    return;
  }
  if (cnode_data->isa<abstract::AbstractScalar>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    (void)me_types->emplace_back(me_type);
    return;
  }
  auto abstract_tuple = cnode_data->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  auto elements = abstract_tuple->elements();
  for (size_t i = 0; i < abstract_tuple->size(); ++i) {
    GetMeRetDataType(elements[i], me_types);
  }
}

transform::TensorOrderMap GetDefaultParams(const FuncGraphPtr &anf_graph,
                                           std::map<std::string, ShapeVector> *origin_shape) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      MS_EXCEPTION_IF_NULL(tensor);
      origin_shape->emplace(para->name(), tensor->shape_c());
      // need ref shape when auto parallel
      auto build_shape = para->abstract()->BuildShape();
      if (build_shape != nullptr) {
        (void)tensor->MetaTensor::set_shape(build_shape->cast<abstract::ShapePtr>()->shape());
        MS_LOG(INFO) << "ref abstract Parameter: " << para->name() << ", tensor: " << tensor->ToString();
      }
      res.emplace(para->name(), tensor);
      MS_LOG(DEBUG) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

void RevertOriginShape(const KernelGraphPtr &anf_graph, const std::map<std::string, ShapeVector> &origin_shape) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto it = origin_shape.find(para->name());
      if (it == origin_shape.end()) {
        MS_LOG(ERROR) << "Failed to find input " << para->name() << " in input_shape " << origin_shape;
        continue;
      }
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      (void)tensor->MetaTensor::set_shape(it->second);
      MS_LOG(INFO) << "ref abstract Parameter: " << para->name() << ", tensor: " << tensor->ToString();
    }
  }
}

std::vector<transform::GeTensorPtr> GetInputTensors(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap init_input_map;
  std::vector<tensor::TensorPtr> init_input;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      (void)init_input_map.emplace(para->name(), value->cast<std::shared_ptr<tensor::Tensor>>());
    }
  }
  (void)std::transform(init_input_map.begin(), init_input_map.end(), std::back_inserter(init_input),
                       [](const std::pair<std::string, tensor::TensorPtr> &item) { return item.second; });
  return transform::ConvertInputTensors(init_input, kOpFormat_NCHW);
}

void RunGEInitGraph(const FuncGraphPtr &anf_graph) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  MS_EXCEPTION_IF_NULL(anf_graph);

  transform::RunOptions run_options;
  run_options.name = "init_subgraph." + anf_graph->ToString();

  auto graph_runner = transform::CheckAndGetGraphRunner(run_options);
  if (graph_runner == nullptr) {
    return;
  }

  std::vector<transform::GeTensorPtr> ge_tensors;
  std::vector<transform::GeTensorPtr> ge_outputs;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    transform::Status ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << run_options.name << " graph failed.";
    }

    MS_LOG(DEBUG) << "Exec " << run_options.name << " graph success.";

    if ((ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::DISTRIBUTION) &&
        (transform::GetGraphByName(BROADCAST_GRAPH_NAME) != nullptr)) {
      run_options.name = BROADCAST_GRAPH_NAME;
      ge_tensors = GetInputTensors(anf_graph);
      ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
      if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec BROADCAST_GRAPH_NAME failed.";
      }
      MS_LOG(DEBUG) << "Exec broadcast graph success.";
    }
  }
}

void UpdateOutputNodeShape(const AnfNodePtr &node, size_t index, TypeId output_type, const ShapeVector &output_shape) {
  MS_EXCEPTION_IF_NULL(node);
  std::string name;
  if (node->isa<CNode>()) {
    name = common::AnfAlgo::GetCNodeName(node);
  }
  size_t total_output_num = AnfAlgo::GetOutputElementNum(node);
  if (index >= total_output_num) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid output index " << index << ", node " << node->fullname_with_scope()
                                      << " has " << total_output_num << " outputs.";
  }
  std::vector<TypeId> types = {};
  std::vector<ShapeVector> shapes = {};
  for (size_t i = 0; i < total_output_num; ++i) {
    if (i == index && kIgnoreGEShapeOps.count(name) == 0) {
      types.push_back(output_type);
      shapes.push_back(output_shape);
    } else {
      types.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
      (void)shapes.emplace_back(common::AnfAlgo::GetOutputInferShape(node, i));
    }
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, node.get());
}

void SetDynamicShapeAttr(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = TopoSort(kernel_graph->output());
  for (auto &node : nodes) {
    if (common::AnfAlgo::IsDynamicShape(node)) {
      MS_LOG(DEBUG) << "Set Dynamic Shape Attr to Node : " << node->fullname_with_scope();
      kernel_graph->SetGraphDynamicAttr(true);
      return;
    }
  }
}

void EnableGraphInputZeroCopy(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // Zero copy is only enabled for PyNative and Subgraph sink.
  if ((!graph->has_flag(kFlagPyNativeRunInGraph) && !graph->has_flag(kFlagEnableZeroCopyInGraph)) ||
      !graph->is_graph_run_mode()) {
    return;
  }
  const auto &input_nodes = graph->input_nodes();
  for (const auto &input : input_nodes) {
    MS_EXCEPTION_IF_NULL(input);
    if (AnfAlgo::OutputAddrExist(input, 0)) {
      auto input_address = AnfAlgo::GetMutableOutputAddr(input, 0, false);
      MS_EXCEPTION_IF_NULL(input_address);
      input_address->set_is_ptr_persisted(false);
      input_address->ClearFlag(device::kDeviceAddressFlagNotUsed);
      MS_LOG(INFO) << "Enable zero copy for input " << input->DebugString();
    }
  }
}

void EnableGraphOutputZeroCopy(const KernelGraphPtr &graph) {
  MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy start";
  MS_EXCEPTION_IF_NULL(graph);
  if ((!graph->has_flag(kFlagEnableZeroCopyInGraph)) || !graph->is_graph_run_mode()) {
    MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy start return";
    return;
  }
  // Zero copy is only enabled for subgraph sink.
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output : outputs) {
    const auto &node_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    const auto &node = node_with_index.first;
    const auto &index = node_with_index.second;
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy check node:" << node->DebugString();
    if (node->isa<CNode>() && AnfAlgo::OutputAddrExist(node, index)) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(node, index, false);
      MS_EXCEPTION_IF_NULL(device_address);
      device_address->set_is_ptr_persisted(false);
      MS_LOG(DEBUG) << "Disable ptr persisted in output node:" << node->DebugString() << " index:" << index
                    << " address:" << device_address << " for graph:" << graph->ToString();
    }
  }
}

struct GraphSummary {
  size_t const_memory_size = 0;
  size_t feature_memory_size = 0;
  bool is_feature_memory_refreshable = false;
  size_t stream_num = 0;
  size_t event_num = 0;
  std::vector<ShapeVector> output_shapes = {};
  std::vector<ge::DataType> output_dtypes = {};
  // pair<input_index, output_index>
  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  bool is_static = false;

  GraphSummary() = default;
  explicit GraphSummary(const ::ge::CompiledGraphSummaryPtr &graph_summary) {
    MS_EXCEPTION_IF_NULL(graph_summary);
    is_static = graph_summary->IsStatic();
    if (is_static) {
      ::ge::graphStatus status;
      status = graph_summary->GetConstMemorySize(const_memory_size);
      if (status != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetConstMemorySize failed, status = " << status;
      }
      status = graph_summary->GetFeatureMemorySize(feature_memory_size);
      if (status != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetFeatureMemorySize failed, status = " << status;
      }
      status = graph_summary->GetFeatureMemoryBaseRefreshable(is_feature_memory_refreshable);
      if (status != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetFeatureMemoryBaseRefreshable failed, status = " << status;
      }
      status = graph_summary->GetStreamNum(stream_num);
      if (status != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetStreamNum failed, status = " << status;
      }
      status = graph_summary->GetEventNum(event_num);
      if (status != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetEventNum failed, status = " << status;
      }
      std::vector<::ge::Shape> ge_shapes;
      status = graph_summary->GetOutputShapes(ge_shapes);
      if (status != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetOutputShapes failed, status = " << status;
      }
      (void)std::transform(ge_shapes.begin(), ge_shapes.end(), std::back_inserter(output_shapes),
                           [](const ::ge::Shape &ge_shape) -> ShapeVector { return ge_shape.GetDims(); });
      if (graph_summary->GetOutputDtypes(output_dtypes) != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetOutputDtypes failed, status = " << status
                          << ", maybe the execution mode is not as expected.";
      }
      if (graph_summary->GetIOIndexesWithSameAddr(io_indexes) != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "GetIOIndexesWithSameAddr failed, status = " << status
                          << ", maybe the execution mode is not as expected.";
      }
    } else {
      MS_LOG(WARNING) << "Graph is not static, maybe the execution mode is not as expected.";
    }
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "const_memory_size[" << const_memory_size << "], feature_memory_size[" << feature_memory_size
       << "], is_feature_memory_refreshable[" << is_feature_memory_refreshable << "], stream_num[" << stream_num
       << "], event_num[" << event_num << "], output size[" << output_shapes.size() << "], is_static[" << is_static
       << "]";
    if (!output_shapes.empty()) {
      if (output_shapes.size() != output_dtypes.size()) {
        MS_LOG(WARNING) << "The output_dtypes size in summary is not equal to output_shapes size.";
      }
      for (size_t i = 0; i < output_shapes.size(); ++i) {
        std::string shape_str = "[";
        std::string dtype_str = "";
        for (size_t j = 0; j < output_shapes[i].size(); ++j) {
          if (j != output_shapes[i].size() - 1) {
            shape_str += std::to_string(output_shapes[i][j]) + ",";
          } else {
            shape_str += std::to_string(output_shapes[i][j]) + "]";
          }
        }

        if (output_shapes[i].empty()) {
          shape_str = "[]";
        }
        if (i < output_dtypes.size()) {
          dtype_str += "[";
          dtype_str += TransGeDtypeToString(output_dtypes[i]);
          dtype_str += "]";
        }
        if (dtype_str.empty()) {
          ss << ", output[" << i << "] shape = " << shape_str;
        } else {
          ss << ", output[" << i << "] shape = " << shape_str << " dtype = " << dtype_str;
        }
      }
    }
    if (!io_indexes.empty()) {
      std::string io_indexes_str = "[";
      for (auto io_index : io_indexes) {
        io_indexes_str += "[" + std::to_string(io_index.first) + "," + std::to_string(io_index.second) + "]";
      }
      io_indexes_str += "]";
      ss << ", io_indexes: " << io_indexes_str;
    }

    return ss.str();
  }

 private:
  std::string TransGeDtypeToString(const transform::GeDataType dtype) const {
    std::string dtype_str = "";
    if (transform::ge_dtype_str_map.find(dtype) != transform::ge_dtype_str_map.end()) {
      dtype_str = transform::ge_dtype_str_map[dtype];
    }
    return dtype_str;
  }
};

std::multimap<std::string, ParameterPtr> FilterAllParameters(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::multimap<std::string, ParameterPtr> ret;
  std::vector<AnfNodePtr> todo = kernel_graph->input_nodes();
  (void)todo.insert(todo.end(), kernel_graph->child_graph_result().begin(), kernel_graph->child_graph_result().end());
  for (const auto &node : todo) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto parameter = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    std::string name = parameter->name();
    (void)ret.emplace(name, parameter);
  }
  return ret;
}

void SetParameterKernelInfo(const AnfNodePtr &node, const std::shared_ptr<device::KernelInfo> &kernel_info) {
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (!build_info) {
    MS_LOG(ERROR) << "Parameter doesn't have build info: " << node->DebugString()
                  << ", full name: " << node->fullname_with_scope();
    return;
  }
  std::vector<TypeId> refresh_output_types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  build_info->SetOutputsDeviceType(refresh_output_types);
}

void SetKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // If kernel build info has been set up. skip
  std::shared_ptr<device::KernelInfo> kernel_info =
    std::dynamic_pointer_cast<device::KernelInfo>(node->kernel_info_ptr());
  if (utils::isa<ParameterPtr>(node)) {
    SetParameterKernelInfo(node, kernel_info);
    return;
  }

  if (!kernel_info) {
    kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    node->set_kernel_info(kernel_info);
  }

  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (!build_info) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    build_info = builder->Build();
  }

  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(node);
  std::vector<TypeId> output_infer_types;
  std::vector<std::string> output_formats;
  for (const auto &output_with_index : output_with_indexs) {
    (void)output_infer_types.emplace_back(
      common::AnfAlgo::GetOutputInferDataType(output_with_index.first, output_with_index.second));
    (void)output_formats.emplace_back(kOpFormat_DEFAULT);
  }
  build_info->SetOutputsDeviceType(output_infer_types);
  build_info->SetOutputsFormat(output_formats);
  kernel_info->set_select_kernel_build_info(build_info);
}

std::string RemoveSuffix(const std::string &str, const std::string &suffix) {
  if (str.size() >= suffix.size() && str.substr(str.size() - suffix.size()) == suffix) {
    return str.substr(0, str.length() - suffix.length());
  }
  return str;
}

bool BuildFakeGraph(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_before_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_before_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif
  (void)setenv("GE_TRAIN", IsGeTrain() ? "1" : "0", 1);
  if (!AddFakeGraph(anf_graph)) {
    MS_LOG(ERROR) << "Add fake graph failed";
    return false;
  }
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_after_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_after_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif
  return true;
}

void ClearForwardOutputAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->has_flag(kFlagPyNativeRunInGraph)) {
    return;
  }
  const auto &input_nodes = graph->input_nodes();
  for (const auto &input : input_nodes) {
    MS_EXCEPTION_IF_NULL(input);
    auto parameter = input->cast<ParameterPtr>();
    if (parameter != nullptr) {
      if (parameter->has_user_data(kForwardOutput)) {
        auto device_address = AnfAlgo::GetMutableOutputAddr(parameter, 0);
        auto new_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
        AnfAlgo::SetOutputAddr(new_address, 0, parameter.get());
        MS_LOG(DEBUG) << "Clear old address " << device_address.get() << " and set new address " << new_address.get()
                      << " to parameter " << parameter->name();
      }
    }
  }
}

class ContextReset {
 public:
  explicit ContextReset(DeviceContext *device_context) : device_context_(device_context) {}
  ~ContextReset() {
    if (device_context_ != nullptr && device_context_->device_res_manager_ != nullptr) {
      device_context_->device_res_manager_->BindDeviceToCurrentThread(true);
    }
  }

 private:
  DeviceContext *device_context_;
};

void UpdateTracker(const std::string &task_name, const std::string &node_name, const std::string &graph_str,
                   size_t size, void *device_ptr, device::tracker::MemType mem_type) {
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, task_name, node_name, graph_str);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, task_name, size, device_ptr, mem_type);
}

void UpdateFMTracker(size_t feature_memory_size, const std::string &graph_name) {
  device::tracker::CALL_MEMORY_TRACKER(AllocMemBlock, 0, feature_memory_size, "Ascend",
                                       AscendMemAdapter::GetInstance().GetActualPeakMemory(), 0, 0, 0);
  device::tracker::CALL_MEMORY_TRACKER(FreeMemBlock, 0, 0, 0);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "RunGeGraph", "RunGeGraph", graph_name);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "RunGeGraph", feature_memory_size, 0,
                                                 device::tracker::MemType::kGeFeatureMemory);
}

bool CacheFileExists(const std::string &name) {
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto dep_files_hash = compile_cache_context.CompileCacheDepFilesHash();
  auto ge_graph_key = name;
  if (!dep_files_hash.empty()) {
    ge_graph_key = dep_files_hash + "_" + ge_graph_key;
  }
  auto ge_cache_path = Common::GetCompilerCachePath() + kGeCache;
  ge_graph_key = NormalizeString(ge_graph_key);
  auto cache_idx_file = ge_cache_path + "/" + ge_graph_key + ".idx";
  struct stat buffer;
  bool ret = stat(cache_idx_file.c_str(), &buffer) == 0;
  MS_LOG(INFO) << "Cached index file name: " << cache_idx_file << " exists: " << ret;
  return ret;
}

}  // namespace

void GeGraphExecutor::AllocInputHostMemory(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &inputs = kernel_graph->inputs();
  auto device_id = device_context_->device_context_key().device_id_;
  for (const auto &input : inputs) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    builder->SetOutputsFormat({kOpFormat_DEFAULT});
    std::vector<TypeId> output_type = {common::AnfAlgo::GetOutputInferDataType(input, 0)};
    builder->SetOutputsDeviceType(output_type);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), input.get());
  }

  for (const auto &input_node : inputs) {
    if (!input_node->isa<Parameter>()) {
      MS_LOG(DEBUG) << input_node->fullname_with_scope() << " is not parameter, continue";
      continue;
    }
    TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(input_node, 0);

    size_t tensor_size;
    if (kernel_graph->is_dynamic_shape()) {
      tensor_size = 0;
    } else {
      std::vector<size_t> shape = Convert2SizeT(common::AnfAlgo::GetOutputInferShape(input_node, 0));
      size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
      tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    }

    auto input_with_index = std::make_pair(input_node, 0);
    const auto kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      input_with_index, nullptr, tensor_size, kOpFormat_DEFAULT, output_type_id, {}, kAscendDevice, device_id);
    auto device_address_ptr = std::make_shared<GeHostAddress>(kernel_tensor);
    device_address_ptr->set_is_ptr_persisted(false);
    AnfAlgo::SetOutputAddr(device_address_ptr, 0, input_node.get());
  }
}

void GeGraphExecutor::AllocOutputHostMemory(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  auto device_id = device_context_->device_context_key().device_id_;
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    SetKernelInfo(output_node);

    // Parameter's memory is allocated earlier, and there is no need to reallocate memory if Parameter is output.
    if (output_node->isa<Parameter>()) {
      continue;
    }

    auto i = output_with_index.second;
    TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(output_node, i);
    const auto kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      output_with_index, nullptr, 0, kOpFormat_DEFAULT, output_type_id, {}, kAscendDevice, device_id);
    auto output_device_addr = std::make_shared<GeHostAddress>(kernel_tensor);
    AnfAlgo::SetOutputAddr(output_device_addr, i, output_node.get());

    if (common::AnfAlgo::IsNopNode(output_node)) {
      auto [real_node, real_idx] = common::AnfAlgo::GetPrevNodeOutput(output_node, i, true);
      if (real_node != output_node || real_idx != i) {
        // set output addr size if the input node is output.
        const auto &inputs = kernel_graph->inputs();
        if (std::any_of(inputs.begin(), inputs.end(),
                        [&real_node](const AnfNodePtr &input_node) { return real_node == input_node; })) {
          auto real_node_addr = AnfAlgo::GetMutableOutputAddr(real_node, real_idx);
          output_device_addr->SetSize(real_node_addr->GetSize());
        }
        AnfAlgo::SetOutputAddr(output_device_addr, real_idx, real_node.get());
      }
    }
  }
}

void GeGraphExecutor::AllocConstMemory(const transform::RunOptions &options, const KernelGraphPtr &graph,
                                       size_t memory_size) const {
  if (memory_size == 0) {
    return;
  }
  MS_LOG(INFO) << "Start AllocConstMemory, memory_size: " << memory_size;
  auto memory = ResManager()->AllocateMemory(memory_size);
  if (memory == nullptr) {
    MS_LOG(EXCEPTION) << "Allocate memory failed, memory size:" << memory_size << ", graph: " << graph->ToString();
  }
  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: ConstMemory, size: " << memory_size
                    << ", graph: " << graph->ToString() << ", device address addr: " << memory;
  }
  UpdateTracker("AllocConstMemory", "ConstMemory", graph->ToString(), memory_size, memory,
                device::tracker::MemType::kGeConst);
  auto graph_runner = transform::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  auto ret = graph_runner->SetConstMemory(options, memory, memory_size);
  if (ret != transform::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "SetConstMemory for graph " << options.name << " failed.";
  }
  MS_LOG(INFO) << "End AllocConstMemory";
}

void GeGraphExecutor::AllocFeatureMemory(const transform::RunOptions &options, size_t memory_size) const {
  if (memory_size == 0) {
    return;
  }
  MS_LOG(INFO) << "Start AllocFeatureMemory, memory_size: " << memory_size;
  auto memory_manager = ResManager()->mem_manager_;
  MS_EXCEPTION_IF_NULL(memory_manager);
  memory_manager->ResetDynamicMemory();
  auto memory = memory_manager->MallocWorkSpaceMem(memory_size);
  if (memory == nullptr) {
    MS_LOG(EXCEPTION) << "AllocFeatureMemory error, memory not enough, memory size: " << memory_size;
  }
  auto graph_runner = transform::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  auto ret = graph_runner->UpdateFeatureMemory(options, memory, memory_size);
  if (ret != transform::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "UpdateFeatureMemory for graph " << options.name << " failed.";
  }
  memory_manager->ResetDynamicMemory();
  MS_LOG(INFO) << "End AllocFeatureMemory";
}

void GeGraphExecutor::AllocParameterMemory(const KernelGraphPtr &kernel_graph, std::set<KernelGraphPtr> *memo) const {
  // Set Device Type to be same as Host Type, AssignStaticMemoryInput will ignore parameters without DeviceType
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (memo == nullptr) {
    MS_LOG(INFO) << "Start AllocParameterMemory, kernel graph: " << kernel_graph->ToString();
    std::set<KernelGraphPtr> memo_set;
    AllocParameterMemory(kernel_graph, &memo_set);
    MS_LOG(INFO) << "AllocParameterMemory finish.";
    return;
  } else if (memo->find(kernel_graph) != memo->end()) {
    return;
  }
  (void)memo->insert(kernel_graph);
  auto parameters = FilterAllParameters(kernel_graph);
  for (const auto &iter : parameters) {
    auto parameter = utils::cast<ParameterPtr>(iter.second);
    if (parameter == nullptr) {
      continue;
    }
    SetKernelInfo(parameter);
  }
  runtime::DeviceAddressUtils::CreateParameterDeviceAddress(device_context_, kernel_graph);
  // call AssignStaticMemoryInput recursively
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(*kernel_graph.get());
}

void GeGraphExecutor::BuildInputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_LOG(INFO) << "Start BuildInputDataGeTensor, kernel graph: " << kernel_graph->ToString();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_inputs;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> need_update_input;
  InputNameAndType input_names;
  auto input_name_list = kernel_graph->user_data<transform::InputNameList>();
  if (input_name_list) {
    input_names = input_name_list->input_names;
  }
  if (input_names.empty()) {
    MS_LOG(INFO) << "Kernel graph: " << kernel_graph->graph_id() << " input data list is nullptr";
    input_datas_[kernel_graph.get()] = {ge_inputs, need_update_input};
    return;
  }
  auto parameters = FilterAllParameters(kernel_graph);
  const auto &cur_inputs = kernel_graph->get_inputs();
  size_t cur_inputs_index = 0;
  for (auto [name, is_ref] : input_names) {
    AnfNodePtr node = nullptr;
    if (!is_ref) {
      while (HasAbstractMonad(cur_inputs.at(cur_inputs_index))) {
        cur_inputs_index++;
      }
      auto abs = cur_inputs.at(cur_inputs_index)->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      while (abs->isa<abstract::AbstractSequence>()) {
        cur_inputs_index++;
        abs = cur_inputs.at(cur_inputs_index)->abstract();
        MS_EXCEPTION_IF_NULL(abs);
      }
      node = cur_inputs.at(cur_inputs_index);
      cur_inputs_index++;
    } else {
      auto iter = parameters.find(name);
      if (iter == parameters.end()) {
        MS_LOG(WARNING) << "Cannot find parameter " << name << " from kernel graph: " << kernel_graph->graph_id();
        name = RemoveSuffix(name, "_temp");
        iter = parameters.find(name);
      }
      if (iter != parameters.end()) {
        node = iter->second;
      } else {
        MS_LOG(EXCEPTION) << "Cannot find parameter " << name << " from kernel graph: " << kernel_graph->graph_id();
      }
    }
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(INFO) << "Build input ge tensor: " << name << ", kernel graph: " << kernel_graph->graph_id();
    auto output_addr = AnfAlgo::GetMutableOutputAddr(node, 0, false);
    auto shapes = trans::GetRuntimePaddingShape(node, 0);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
    auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
    GeTensor ge_tensor(*ge_tensor_desc);
    if (output_addr->GetMutablePtr() != nullptr) {
      if (ge_tensor.SetData(reinterpret_cast<uint8_t *>(output_addr->GetMutablePtr()), output_addr->GetSize(),
                            [](void *) {}) != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "SetData failed, ge input data " << ge_inputs.size() << " name: " << name
                          << " size: " << output_addr->GetSize();
      }
      if (kernel_graph->is_dynamic_shape()) {
        (void)need_update_input.emplace_back(node, ge_inputs.size());
      }
      MS_LOG(INFO) << "ge input data " << ge_inputs.size() << " name: " << name << " size: " << output_addr->GetSize();
    }
    // The device address of input tensor may change every step.
    // Always keep the input node address consistent with the input tensor address.
    (void)need_update_input.emplace_back(node, ge_inputs.size());
    (void)ge_inputs.emplace_back(std::move(ge_tensor));
  }
  while (cur_inputs_index < cur_inputs.size() && HasAbstractMonad(cur_inputs.at(cur_inputs_index))) {
    cur_inputs_index++;
  }
  if (cur_inputs_index != cur_inputs.size()) {
    MS_LOG(WARNING) << "Not use all cur inputs, cur_inputs_index: " << cur_inputs_index
                    << ", cur_inputs.size(): " << cur_inputs.size() << ", kernel graph: " << kernel_graph->graph_id();
  }
  input_datas_[kernel_graph.get()] = {ge_inputs, need_update_input};
  MS_LOG(INFO) << "BuildInputDataGeTensor finish.";
}

void GeGraphExecutor::BuildOutputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_LOG(INFO) << "Start BuildOutputDataGeTensor, kernel graph: " << kernel_graph->ToString();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_outputs;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> graph_outputs;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    auto index = output_with_index.second;
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    if (common::AnfAlgo::IsNoOuputNode(output_node)) {
      continue;
    }
    auto real_index = output_node->isa<ValueNode>() ? 0 : index;
    auto shapes = trans::GetRuntimePaddingShape(output_node, real_index);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
    auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
    GeTensor ge_tensor(*ge_tensor_desc);
    (void)ge_outputs.emplace_back(std::move(ge_tensor));
    (void)graph_outputs.emplace_back(output_node, index);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(
    ge_outputs.size() == graph_outputs.size(),
    "The size of ge_outputs and graph_outputs check error, kernel graph: " + kernel_graph->ToString());
  output_datas_[kernel_graph.get()] = {ge_outputs, graph_outputs};
  MS_LOG(INFO) << "BuildOutputDataGeTensor finish.";
}

DeviceAddressPtr GeGraphExecutor::CreateOutputDeviceAddress(const KernelGraphPtr &kernel_graph,
                                                            const KernelWithIndex &output_with_index,
                                                            size_t need_alloc_output_cnt) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto output_node = output_with_index.first;
  MS_EXCEPTION_IF_NULL(output_node);
  auto ref_map = kernel_graph->GetRefMap();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto real_index = output_node->isa<ValueNode>() ? 0 : output_with_index.second;
  TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
  size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
  auto shapes = trans::GetRuntimePaddingShape(output_node, real_index);
  auto tensor_size =
    shapes.empty() ? type_size : std::accumulate(shapes.begin(), shapes.end(), type_size, std::multiplies<size_t>());
  // When ValueNode is a graph output, runtime does not manage this memory
  // output in ref_map, mem same is input
  bool need_not_alloc = (kernel_graph->has_flag(kFlagEnableZeroCopyInGraph) && !output_node->isa<ValueNode>()) ||
                        (ref_map.find(output_with_index) != ref_map.end());
  void *mem = need_not_alloc ? nullptr : ResManager()->AllocateMemory(tensor_size);

  if (common::IsNeedProfileMemory() && !need_not_alloc) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: ValueNodeOutput, size:" << tensor_size
                    << ", graph: " << kernel_graph->ToString() << ", node: " << output_node->fullname_with_scope()
                    << ", device address addr: " << mem;
  }
  if (!need_not_alloc) {
    UpdateTracker("ValueNodeOutput", output_node->fullname_with_scope(), kernel_graph->ToString(), tensor_size, mem,
                  device::tracker::MemType::kConstantValue);
  }

  const auto kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {output_node, real_index}, mem, tensor_size, kOpFormat_DEFAULT, output_type_id, {}, kAscendDevice, device_id);
  auto output_device_addr = std::make_shared<AscendDeviceAddress>(kernel_tensor);
  if (ref_map.find(output_with_index) != ref_map.end()) {
    auto input_with_index = ref_map[output_with_index];
    auto input_device_address = AnfAlgo::GetMutableOutputAddr(input_with_index.first, input_with_index.second, false);
    MS_EXCEPTION_IF_NULL(input_device_address);
    MS_LOG(INFO) << "The output node " << output_node->fullname_with_scope()
                 << " is in ref_map, set the same device_address ptr as the corresponding input, input node: "
                 << input_with_index.first->fullname_with_scope();
    // Update the reference count of device address.
    output_device_addr->set_pointer_ref_count(input_device_address->pointer_ref_count());
    output_device_addr->IncreaseOriginalRefCount();
    output_device_addr->ResetRefCount();
  }
  output_device_addr->set_device_synchronizer(std::make_shared<AscendDeviceSynchronizer>());
  output_device_addr->set_is_ptr_persisted(true);
  if (IsMemoryPoolRecycle() && need_alloc_output_cnt <= kNeedRecycleOutput) {
    MS_LOG(INFO) << "Set Memory Pool Recycle, graph: " << kernel_graph->ToString()
                 << ", node: " << output_node->fullname_with_scope();
    output_device_addr->set_from_persistent_mem(true);
    output_device_addr->set_need_recycle(true);
  }
  return output_device_addr;
}

void GeGraphExecutor::AllocOutputMemory(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start AllocOutputMemory, kernel graph: " << kernel_graph->ToString();

  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  auto ref_map = kernel_graph->GetRefMap();
  size_t need_alloc_output_cnt = 0;
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    if (output_node->isa<Parameter>() || output_node->isa<ValueNode>()) {
      continue;
    }
    if (ref_map.find(output_with_index) != ref_map.end()) {
      continue;
    }
    need_alloc_output_cnt++;
  }

  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    SetKernelInfo(output_node);

    // Parameter's memory is allocated earlier, and there is no need to reallocate memory if Parameter is output.
    if (AnfAlgo::OutputAddrExist(output_node, output_with_index.second, false) || output_node->isa<Parameter>()) {
      MS_LOG(INFO) << "The device_address of output node " << output_node->fullname_with_scope()
                   << " is already exist, skip.";
      continue;
    }

    auto output_device_addr = CreateOutputDeviceAddress(kernel_graph, output_with_index, need_alloc_output_cnt);
    AnfAlgo::SetOutputAddr(output_device_addr, output_with_index.second, output_node.get());
    MS_LOG(INFO) << "Output node info: (name " << output_node->fullname_with_scope() << ", "
                 << output_node->DebugString() << " ), output size: " << output_device_addr->GetSize()
                 << ", device_address: " << output_device_addr;
    // When both the input and output of NopNode are used as outputs, different memory needs to be allocated for them.
  }
  MS_LOG(INFO) << "AllocOutputMemory finish.";
}

GeDeviceResManager *GeGraphExecutor::ResManager() const {
  MS_EXCEPTION_IF_NULL(device_context_);
  auto res_manager = dynamic_cast<GeDeviceResManager *>(device_context_->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(res_manager);
  return res_manager;
}

void GeGraphExecutor::PreprocessBeforeRun(const KernelGraphPtr &graph) {
  auto ret = CompileGraph(graph, {});
  if (!ret) {
    MS_LOG(EXCEPTION) << "Compile graph fail, graph id: " << graph->graph_id();
  }
}

bool GeGraphExecutor::BuildGraph(const KernelGraphPtr &graph, const transform::TensorOrderMap &tensor_order_map) {
  std::set<KernelGraphPtr> memo;
  GEGraphOptimization::GetInstance().OptimizeGEGraph(graph, &memo);
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto use_compile_cache = compile_cache_context.UseCompileCache();
  auto name = GetGraphName(graph);
  bool has_cache = CacheFileExists(name);
  if (use_compile_cache && has_cache) {
    MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";
    if (!BuildFakeGraph(graph)) {
      return false;
    }
  } else {
    (void)BuildDFGraph(graph, tensor_order_map, false);
  }
  return true;
}

void GeGraphExecutor::AllocMemory(const KernelGraphPtr &graph) {
  AllocParameterMemory(graph);
  AllocOutputMemory(graph);
  BuildInputDataGeTensor(graph);
  BuildOutputDataGeTensor(graph);
  EnableGraphInputZeroCopy(graph);
  EnableGraphOutputZeroCopy(graph);
}

bool GeGraphExecutor::CompileGraph(const KernelGraphPtr &graph,
                                   const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "ge graph executor compile graph " << graph->ToString();
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto use_compile_cache = compile_cache_context.UseCompileCache();
  std::map<std::string, ShapeVector> origin_shape;
  const auto &tensor_order_map = GetDefaultParams(graph, &origin_shape);
  auto name = GetGraphName(graph);
  bool has_cache = CacheFileExists(name);
  if (use_compile_cache && has_cache) {
    MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";
    std::set<KernelGraphPtr> memo;
    GEGraphOptimization::GetInstance().OptimizeGEGraph(graph, &memo);
    if (!BuildFakeGraph(graph)) {
      return false;
    }
  } else {
    (void)BuildGraph(graph, tensor_order_map);
  }
  SetDynamicShapeAttr(graph);
  transform::RunOptions run_options;
  run_options.name = GetGraphName(graph);
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  // create loop var
  RunInitGraph(run_options.name);
  if (graph->is_dynamic_shape()) {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    auto ret = graph_runner->CompileGraph(run_options);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Compile graph " << run_options.name << " failed.";
    }
  } else {
    ::ge::CompiledGraphSummaryPtr ge_graph_summary = nullptr;
    {
      // Release GIL before calling into (potentially long-running) C++ code
      GilReleaseWithCheck gil_release;
      auto ret = graph_runner->CompileGraph(run_options, &ge_graph_summary);
      if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Compile graph " << run_options.name << " failed.";
      }
    }
    GraphSummary summary(ge_graph_summary);
    MS_LOG(INFO) << "Graph " << run_options.name << " summary: " << summary.ToString();
    feature_memorys[run_options.name] = summary.feature_memory_size;
    streams[run_options.name] = summary.stream_num;
    AllocConstMemory(run_options, graph, summary.const_memory_size);
    AllocFeatureMemory(run_options, summary.feature_memory_size);
    AddRefCorrespondPairs(graph, summary.io_indexes);
  }
  AllocMemory(graph);

  graph->set_run_mode(RunMode::kGraphMode);
  graph->set_memory_managed_by_ge(true);
  if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
    graph->set_is_loop_count_sink(true);
  }
  RevertOriginShape(graph, origin_shape);
  return true;
}

void GeGraphExecutor::AddRefCorrespondPairs(const KernelGraphPtr &graph,
                                            const std::vector<std::pair<uint32_t, uint32_t>> &io_indexes) const {
  MS_LOG(INFO) << "Start convert io_indexes to ref_map, kernel graph: " << graph->ToString();
  MS_EXCEPTION_IF_NULL(graph);

  std::map<session::AnfWithOutIndex, session::AnfWithOutIndex> ref_out_in_map = {};
  auto graph_inputs_all = graph->parameters();
  std::vector<AnfNodePtr> graph_inputs = {};
  for (auto &node : graph_inputs_all) {
    MS_EXCEPTION_IF_NULL(node);
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (HasAbstractMonad(node) || abs->isa<abstract::AbstractSequence>()) {
      MS_LOG(INFO) << "Input node: " << node->DebugString() << " is a monad or tuple/list parameter, skip.";
      continue;
    }
    graph_inputs.emplace_back(node);
  }

  std::vector<common::KernelWithIndex> graph_outputs_all = {};
  common::AnfAlgo::GetRealInputs(graph->get_return(), &graph_outputs_all);
  std::vector<common::KernelWithIndex> graph_outputs = {};

  for (auto &node_with_index : graph_outputs_all) {
    if (common::AnfAlgo::IsNoOuputNode(node_with_index.first) || HasAbstractMonad(node_with_index.first)) {
      MS_LOG(INFO) << "Output node: " << node_with_index.first->fullname_with_scope()
                   << " is a no output node or monad node, skip.";
      continue;
    }

    graph_outputs.emplace_back(node_with_index);
  }

  for (auto in_out_index : io_indexes) {
    if (in_out_index.first >= graph_inputs.size() || in_out_index.second >= graph_outputs.size()) {
      MS_LOG(EXCEPTION) << "The io_indexes out of range, input index: " << in_out_index.first
                        << ", output index: " << in_out_index.second << ", graph input size: " << graph_inputs.size()
                        << ", graph output size: " << graph_outputs.size();
    }
    session::AnfWithOutIndex origin_node = std::make_pair(graph_inputs[in_out_index.first], 0);
    session::AnfWithOutIndex final_node = graph_outputs[in_out_index.second];
    if (origin_node.first == final_node.first) {
      MS_LOG(INFO) << "The origin node is same as final node, node: " << origin_node.first->fullname_with_scope();
      continue;
    }
    if (ref_out_in_map.count(final_node) != 0) {
      MS_LOG(INFO) << "The node is already in ref_out_in_map, node: " << final_node.first->fullname_with_scope()
                   << ", index: " << final_node.second;
      continue;
    }
    // if input node is not abstract ref, set ref may cause memory reuse error
    auto abs = origin_node.first->abstract();
    if (!abs->isa<abstract::AbstractRefTensor>()) {
      MS_LOG(INFO) << "The node is not abstract tensor: " << final_node.first->fullname_with_scope()
                   << ", index: " << final_node.second;
      continue;
    }

    ref_out_in_map.emplace(final_node, origin_node);
    MS_LOG(INFO) << "Convert io_index [" << in_out_index.first << ", " << in_out_index.second
                 << "] to ref_out_in_map, final_node: " << final_node.first->fullname_with_scope()
                 << ", index:" << final_node.second << ", origin_node: " << origin_node.first->fullname_with_scope()
                 << ", index: " << origin_node.second;
  }

  graph->set_ref_out_in_map(ref_out_in_map);
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  MS_EXCEPTION_IF_NULL(graph);

  auto graph_name = GetGraphName(graph);
  profiler::CollectHostInfo("Ascend", "CompileGraph", "GeCompileGraph_" + graph_name, 1, 0, kCollectHostInfoStart);

  // cppcheck-suppress unreadVariable
  ContextReset reset_context(device_context_);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kg);
  if (IsEnableRefMode()) {
    auto ret = CompileGraph(kg, compile_options);
    profiler::CollectHostInfo("Ascend", "CompileGraph", "GeCompileGraph_" + graph_name, 1, 0, kCollectHostInfoEnd);
    return ret;
  } else {
    // delete SetCPUMemManager when delete env MS_DISABLE_REF_MODE
    ResManager()->SetCPUMemManager();
    std::map<std::string, ShapeVector> origin_shape;
    const auto &tensor_order_map = GetDefaultParams(graph, &origin_shape);
    auto &compile_cache_context = CompileCacheContext::GetInstance();
    auto use_compile_cache = compile_cache_context.UseCompileCache();
    if (use_compile_cache) {
      MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";
      std::set<KernelGraphPtr> memo;
      GEGraphOptimization::GetInstance().OptimizeGEGraph(kg, &memo);
      if (!BuildFakeGraph(kg)) {
        profiler::CollectHostInfo("Ascend", "CompileGraph", "GeCompileGraph_" + graph_name, 1, 0, kCollectHostInfoEnd);
        return false;
      }
    } else {
      (void)BuildGraph(kg, tensor_order_map);
    }
    SetDynamicShapeAttr(kg);
    AllocInputHostMemory(kg);
    AllocOutputHostMemory(kg);
    kg->set_run_mode(RunMode::kGraphMode);
    if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
      kg->set_is_loop_count_sink(true);
    }
    // copy init weight to device
    RunGEInitGraph(kg);
    RevertOriginShape(kg, origin_shape);
    profiler::CollectHostInfo("Ascend", "CompileGraph", "GeCompileGraph_" + graph_name, 1, 0, kCollectHostInfoEnd);
    return true;
  }
}

void SetOutputs(const std::vector<KernelWithIndex> &graph_outputs,
                const std::vector<transform::GeTensorPtr> &ge_outputs, const std::vector<TypeId> &me_types) {
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    const auto &tensor = ge_outputs[i];
    auto output_addr = AnfAlgo::GetMutableOutputAddr(output_node, idx);
    ::ge::Placement dp = tensor->GetTensorDesc().GetPlacement();
    auto &&ge_data_uni = tensor->ResetData();
    auto deleter = ge_data_uni.get_deleter();
    auto ge_data = ge_data_uni.release();
    MS_EXCEPTION_IF_NULL(ge_data);
    if (dp == ::ge::kPlacementHost) {
      constexpr int64_t kTensorAlignBytes = 64;
      if (reinterpret_cast<uintptr_t>(ge_data) % kTensorAlignBytes != 0) {
        MS_LOG(EXCEPTION) << "Skip zero-copy ge tensor " << reinterpret_cast<uintptr_t>(ge_data)
                          << ", bytes not aligned with expected.";
      }
      if (me_types[i] == TypeId::kObjectTypeString) {
        MS_LOG(EXCEPTION) << "It is not supported that Output node " << output_node->DebugString()
                          << "'s output data type is string now.";
      }
      MS_LOG(DEBUG) << "Zero-copy ge tensor " << reinterpret_cast<uintptr_t>(ge_data) << " as aligned with "
                    << kTensorAlignBytes << " types.";
      output_addr->set_is_ptr_persisted(false);
      output_addr->set_from_mem_pool(false);
      output_addr->set_deleter(deleter);
      output_addr->set_ptr(ge_data);
      output_addr->SetSize(tensor->GetSize());
    } else {
      MS_LOG(EXCEPTION) << "It is not supported that Output node " << output_node->DebugString()
                        << "'s output data's placement is device now.";
    }
    auto actual_shapes = tensor->GetTensorDesc().GetShape().GetDims();
    UpdateOutputNodeShape(output_node, idx, me_types[i], actual_shapes);
  }
}

void SetOutput(GeDeviceResManager *res_manager, GeTensor *ge_output, const AnfNodePtr &output_node, size_t idx) {
  if (output_node->isa<ValueNode>()) {
    auto &&ge_data_uni = ge_output->ResetData();
    auto deleter = ge_data_uni.get_deleter();
    auto ge_data = ge_data_uni.release();
    deleter(ge_data);
    return;
  }
  auto actual_shapes = ge_output->GetTensorDesc().GetShape().GetDims();
  for (size_t i = 0; i < actual_shapes.size(); ++i) {
    if (actual_shapes[i] < 0) {
      MS_LOG(EXCEPTION) << "Output shape must be greater than 0, but got " << actual_shapes;
    }
  }
  auto output_addr = AnfAlgo::GetMutableOutputAddr(output_node, idx, false);
  output_addr->SetSize(ge_output->GetSize());
  auto &&ge_data_uni = ge_output->ResetData();
  auto deleter = ge_data_uni.get_deleter();
  auto ge_data = ge_data_uni.release();
  MS_EXCEPTION_IF_NULL(ge_data);
  output_addr->set_is_ptr_persisted(false);
  output_addr->set_from_mem_pool(false);
  output_addr->set_deleter(deleter);
  output_addr->set_ptr(ge_data);
  auto placement = ge_output->GetTensorDesc().GetPlacement();
  if (placement == ::ge::kPlacementHost) {
    MS_LOG(DEBUG) << output_node->DebugString() << "'s output data's placement is host";
    size_t size = ge_output->GetSize();
    void *mem = res_manager->AllocateMemory(size);
    if (mem == nullptr) {
      MS_LOG(EXCEPTION) << "Allocate memory failed, memory size:" << size
                        << ", output_node: " << output_node->ToString();
    }
    output_addr->set_from_mem_pool(true);
    output_addr->set_ptr(mem);
    auto *ascend_addr = dynamic_cast<AscendDeviceAddress *>(output_addr.get());
    MS_EXCEPTION_IF_NULL(ascend_addr);
    ascend_addr->SyncHostToDevice(size, ge_data);
  }
  // Update shape in kernel tensor.
  const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, idx);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  kernel_tensor->SetShapeVector(actual_shapes);
  MS_LOG(INFO) << "[ZeroCopy] Update output " << output_node->DebugString() << " address to "
               << output_addr->GetMutablePtr() << ", shape:" << actual_shapes
               << ", type: " << TypeIdToString(output_addr->type_id()) << ", format: " << output_addr->format();
}

void SetDynamicOutputs(const std::vector<KernelWithIndex> &graph_outputs, std::vector<GeTensor> *ge_outputs,
                       GeDeviceResManager *res_manager) {
  MS_EXCEPTION_IF_NULL(res_manager);
  size_t ge_outputs_index = 0;
  size_t ge_outputs_size = ge_outputs->size();
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    if (common::AnfAlgo::IsNoOuputNode(output_node)) {
      continue;
    }
    if (ge_outputs_index >= ge_outputs_size) {
      MS_LOG(EXCEPTION) << "GE data access is out of bounds, which the current index value is " << ge_outputs_index
                        << ", the total number of GE output is " << ge_outputs_size << ".";
    }
    SetOutput(res_manager, &((*ge_outputs)[ge_outputs_index++]), output_node, idx);
  }
}

size_t GeGraphExecutor::GetGraphFeatureMemory(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  auto iter = feature_memorys.find(graph_name);
  if (iter == feature_memorys.end()) {
    MS_LOG(EXCEPTION) << "Graph " << graph_name << " feature memory not found.";
  }
  auto stream_iter = streams.find(graph_name);
  if (stream_iter == streams.end()) {
    MS_LOG(EXCEPTION) << "Graph " << graph_name << " stream not found.";
  }
  MS_LOG(WARNING) << "Need Profile Memory, graph: " << graph_name << ", stream: " << stream_iter->second;
  auto max_static_memory_size = ResManager()->GetMaxUsedMemorySize();
  auto feature_memory_size = iter->second;
  auto total_memory_size = max_static_memory_size + feature_memory_size;
  AscendMemAdapter::GetInstance().UpdateActualPeakMemory(total_memory_size);
  UpdateFMTracker(feature_memory_size, graph_name);
  return feature_memory_size;
}
int64_t GeGraphExecutor::CurGraphSinkSize(std::string graph_name) {
  int64_t sink_size = -1;
  auto result = graph_sink_size_.find(graph_name);
  if (result != graph_sink_size_.end()) {
    sink_size = result->second;
  } else {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE &&
        ms_context->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK)) {
      sink_size = ConfigManager::GetInstance().iter_num();
    }
    MS_LOG(INFO) << "Graph [" << graph_name << "] sink size is " << sink_size;
    graph_sink_size_.insert(std::pair(graph_name, sink_size));
  }
  return sink_size;
}

bool GeGraphExecutor::RunGraphRefMode(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  RunInitGraph(graph_name);
  MS_LOG(INFO) << "GE run graph start in ref mode, graph: " << graph_name << ".";
  (void)ResManager()->BindDeviceToCurrentThread(false);

  // call ge rungraph
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  transform::RunOptions run_options;
  run_options.name = graph_name;
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  std::vector<GeTensor> ge_inputs = GenerateInputGeTensor(kg);
  std::vector<GeTensor> ge_outputs = GenerateOutputGeTensor(kg);

  bool is_dynamic_shape = kg->is_dynamic_shape();
  if (IsMemoryPoolRecycle() && !is_dynamic_shape) {
    auto max_static_memory_size = ResManager()->GetMaxUsedMemorySize();
    auto iter = feature_memorys.find(graph_name);
    if (iter == feature_memorys.end()) {
      MS_LOG(EXCEPTION) << "Graph " << graph_name << " feature memory not found.";
    }
    auto feature_memory_size = iter->second;
    if (feature_memory_size != 0) {
      size_t total_memory_size = max_static_memory_size + feature_memory_size;
      size_t max_hbm_memory_size = static_cast<size_t>(AscendMemAdapter::GetInstance().GetMsUsedHbmSize());
      AscendMemAdapter::GetInstance().UpdateActualPeakMemory(total_memory_size);
      UpdateFMTracker(feature_memory_size, graph_name);
      if (common::IsNeedMemoryStatistic()) {
        MS_LOG(WARNING) << "Now Memory Status, graph: " << graph_name
                        << ", max_static_memory_size: " << max_static_memory_size
                        << ", feature_memory_size: " << feature_memory_size
                        << ", max_hbm_memory_size: " << max_hbm_memory_size;
      }
      if (total_memory_size > max_hbm_memory_size) {
        MS_LOG(EXCEPTION) << "Memory pool not enough, graph: " << graph_name
                          << ", max_static_memory_size: " << max_static_memory_size
                          << ", feature_memory_size: " << feature_memory_size
                          << ", max_hbm_memory_size: " << max_hbm_memory_size;
      }
    }
  }

  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    MS_LOG(INFO) << "Run graph begin, inputs size is: " << inputs.size() << ", " << graph_name;
    transform::Status ret =
      transform::RunGraphWithStreamAsync(graph_runner, run_options, ResManager()->GetStream(), ge_inputs, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec graph failed";
    }
  }
  if (is_dynamic_shape) {
    auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    SetDynamicOutputs(graph_outputs, &ge_outputs, ResManager());
    auto sync_ret = ResManager()->SyncStream();
    if (!sync_ret) {
      MS_LOG(EXCEPTION) << "Sync stream failed";
    }
  }
  ClearForwardOutputAddress(kg, device_context_);
  return true;
}

void GeGraphExecutor::DoAsyncCkpt(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto env = common::GetEnv("MS_ENABLE_CKPT_D2H_ASYNC");
  if (env == "1" && ms_context->get_param<bool>(MS_CTX_NEED_CKPT) && kg != nullptr) {
    auto cur_step = ms_context->get_param<int>(MS_CTX_CUR_STEP_NUM);
    auto save_steps = ms_context->get_param<int>(MS_CTX_SAVE_CKPT_STEPS);
    auto last_triggered_step = ms_context->get_param<int>(MS_CTX_LAST_TRIGGERED_STEP);
    MS_LOG(DEBUG) << "cur_step:" << cur_step << ", save_steps: " << save_steps
                  << ", last_triggered_step:" << last_triggered_step;
    if (cur_step >= (last_triggered_step + save_steps)) {
      if (SkipOrResetCopyAction()) {
        MS_LOG(INFO) << "Enable async d2h copy";
        SavePrevStepWeight(kg->GetRootWeights(), ResManager()->GetCopyDataStream());
      }
      if (kg->has_attr(kIsRefGraph) && GetValue<bool>(kg->get_attr(kIsRefGraph)) && SkipOrResetSyncAction()) {
        MS_LOG(INFO) << "Ref graph sync once action";
        SyncCopyStream(ResManager()->GetCopyDataStream());
      }
    }
  }
}

bool GeGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                               std::vector<tensor::Tensor> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, 1, 0, kCollectHostInfoStart);
  DoAsyncCkpt(graph);
  if (IsEnableRefMode()) {
    if (!RunGraphRefMode(graph, inputs)) {
      profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, 1, 0, kCollectHostInfoEnd);
      return false;
    }
  } else {
    MS_LOG(INFO) << "GE run graph start, graph: " << graph_name << ".";
    (void)ResManager()->BindDeviceToCurrentThread(false);
    // copy input from device to host
    const auto &cur_inputs = graph->get_inputs();
    std::vector<tensor::TensorPtr> input_tensors;
    for (const auto &input : cur_inputs) {
      MS_EXCEPTION_IF_NULL(input);
      auto output_addr = AnfAlgo::GetMutableOutputAddr(input, 0);
      auto shapes = trans::GetRuntimePaddingShape(input, 0);
      auto host_type = common::AnfAlgo::GetOutputInferDataType(input, 0);
      auto tensor = std::make_shared<tensor::Tensor>(host_type, shapes);
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->set_device_address(output_addr, false);
      tensor->data_sync();
      (void)input_tensors.emplace_back(std::move(tensor));
    }
    auto ge_inputs = transform::ConvertInputTensors(input_tensors, kOpFormat_NCHW);

    // call ge rungraph
    KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
    if (kg != nullptr) {
      graph_name = kg->GetFuncGraph()->ToString();
    }
    transform::RunOptions run_options;
    run_options.name = graph_name;
    auto graph_runner = transform::GetGraphRunner();
    if (graph_runner == nullptr) {
      MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
    }

    AnfNodePtr output = graph->get_return()->input(1);
    MS_EXCEPTION_IF_NULL(output);
    std::vector<TypeId> me_types;
    auto output_c = output->cast<CNodePtr>()->abstract();
    // get output node data types
    GetMeRetDataType(output_c, &me_types);
    std::vector<transform::GeTensorPtr> ge_outputs;
    {
      // Release GIL before calling into (potentially long-running) C++ code
      GilReleaseWithCheck gil_release;
      MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size();
      transform::Status ret = transform::RunGraphAsync(graph_runner, run_options, ge_inputs, &ge_outputs);
      MS_LOG(DEBUG) << "Run graph finish, outputs size is: " << ge_outputs.size();
      if (ret == transform::Status::NOT_FOUND) {
        MS_LOG(WARNING) << "The Graph[" << graph_name << "] is not found, skip run it.";
        profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, 1, 0, kCollectHostInfoEnd);
        return true;
      } else if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec graph failed";
      }
    }
    auto no_output = common::AnfAlgo::IsNoOuputNode(output);
    if (!no_output) {
      if (me_types.size() != ge_outputs.size()) {
        MS_LOG(EXCEPTION) << "Invalid output size, me_type's size " << me_types.size() << " tensor size "
                          << ge_outputs.size();
      }
      // copy output from host to device
      auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
      if (graph_outputs.size() != ge_outputs.size()) {
        MS_LOG(EXCEPTION) << "Invalid output size, graph's size " << graph_outputs.size() << " tensor size "
                          << ge_outputs.size();
      }
      SetOutputs(graph_outputs, ge_outputs, me_types);
    }
  }
  if (graph->has_flag(transform::kGraphFlagHasGetNext)) {
    MS_LOG(DEBUG) << "Reset ConfigManager, graph: " << graph_name;
    ConfigManager::GetInstance().ResetConfig();
    ConfigManager::GetInstance().ResetIterNum();
  }
  profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, 1, 0, kCollectHostInfoEnd);
  MS_LOG(INFO) << "GE run graph end.";
  return true;
}

FuncGraphPtr GeGraphExecutor::BuildDFGraph(const FuncGraphPtr &anf_graph,
                                           const transform::TensorOrderMap &init_inputs_map, bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_before_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_before_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif

  if (!AddDFGraph(anf_graph, init_inputs_map, export_air)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_after_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_after_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif

  if (export_air) {
    // export air can't use session->AddGraph, it will cause atc error.
    return anf_graph;
  }

  return anf_graph;
}

std::vector<GeTensor> GeGraphExecutor::GenerateInputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_inputs;
  auto iter = input_datas_.find(kernel_graph.get());
  if (iter == input_datas_.end()) {
    return ge_inputs;
  }
  const auto &input_datas = iter->second.ge_inputs;
  ge_inputs = input_datas;
  for (auto &kv : iter->second.need_update_input) {
    auto input_node = kv.first.lock();
    MS_EXCEPTION_IF_NULL(input_node);
    auto output_addr = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
    MS_EXCEPTION_IF_NULL(output_addr);
    bool is_dynamic_shape = kernel_graph->is_dynamic_shape();
    if (is_dynamic_shape) {
      auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(output_addr->kernel_tensor()->GetShapeVector(),
                                                                      output_addr->type_id(), output_addr->format());
      MS_EXCEPTION_IF_NULL(ge_tensor_desc);
      ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
      (void)ge_inputs[kv.second].SetTensorDesc(*ge_tensor_desc);
    }
    if (output_addr->GetMutablePtr() == nullptr) {
      // alloc static memory for unused inputs
      // error in ge when set nullptr into ge tensor
      std::vector<size_t> shape = Convert2SizeT(common::AnfAlgo::GetOutputInferShape(input_node, 0));
      size_t type_size = GetTypeByte(TypeIdToType(common::AnfAlgo::GetOutputInferDataType(input_node, 0)));
      size_t memory_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>{});
      MS_EXCEPTION_IF_NULL(ResManager());
      auto memory = ResManager()->AllocateMemory(memory_size);
      output_addr->set_ptr(memory);
      output_addr->SetSize(memory_size);
      if (common::IsNeedProfileMemory()) {
        MS_LOG(WARNING) << "Need Profile Memory, alloc type: UnusedInput, size:" << memory_size
                        << ", graph: " << kernel_graph->ToString() << ", node: " << input_node->fullname_with_scope()
                        << ", device address addr: " << memory;
      }
      UpdateTracker("UnusedInput", input_node->fullname_with_scope(), kernel_graph->ToString(), memory_size, memory,
                    device::tracker::MemType::kOther);
    }
    if (kv.second >= ge_inputs.size()) {
      MS_LOG(EXCEPTION) << input_node->DebugString() << ", index: " << kv.second << " is greater than "
                        << ge_inputs.size();
    }
    MS_LOG(DEBUG) << "[ZeroCopy] For Graph " << kernel_graph->ToString() << ", update input "
                  << input_node->DebugString() << " address to " << output_addr->GetMutablePtr()
                  << ", shape:" << output_addr->kernel_tensor()->GetShapeVector()
                  << ", type: " << TypeIdToString(output_addr->type_id()) << ", format: " << output_addr->format()
                  << ", memory size: " << output_addr->GetSize();
    if (output_addr->GetPtr() != ge_inputs[kv.second].GetData() ||
        output_addr->GetSize() != ge_inputs[kv.second].GetSize()) {
      (void)ge_inputs[kv.second].SetData(static_cast<uint8_t *>(output_addr->GetMutablePtr()), output_addr->GetSize(),
                                         [](void *) {});
    }
  }
  return ge_inputs;
}

std::vector<GeTensor> GeGraphExecutor::GenerateOutputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_outputs;
  auto iter = output_datas_.find(kernel_graph.get());
  if (iter == output_datas_.end()) {
    return ge_outputs;
  }
  const auto &output_datas = iter->second.ge_outputs;
  ge_outputs = output_datas;

  bool is_dynamic_shape = kernel_graph->is_dynamic_shape();
  size_t idx = 0;
  for (const auto &output : iter->second.graph_outputs) {
    if (is_dynamic_shape) {
      ge_outputs[idx].SetData(nullptr, 0U, [](void *) {});
      idx++;
      continue;
    }
    auto output_node = output.first.lock();
    auto index = output.second;
    MS_EXCEPTION_IF_NULL(output_node);
    MS_EXCEPTION_IF_CHECK_FAIL(
      idx < ge_outputs.size(),
      "GenerateOutputGeTensor idx is greater equal than ge_outputs size, idx: " + std::to_string(idx) +
        ", ge outputs size: " + std::to_string(ge_outputs.size()) + ", kernel graph: " + kernel_graph->ToString());
    auto output_device_addr = AnfAlgo::GetMutableOutputAddr(output_node, index, false);
    MS_LOG(INFO) << "Output addr " << output_device_addr->GetMutablePtr();
    if (output_device_addr->GetMutablePtr() == nullptr) {
      MS_LOG(EXCEPTION) << "Output " << output_node->fullname_with_scope() << ", index: " << index
                        << " address is nullptr, kernel graph: " << kernel_graph->ToString()
                        << ", addr memory size: " << output_device_addr->GetSize()
                        << "\n Maybe memory is not enough, memory statistics:"
                        << AscendMemAdapter::GetInstance().DevMemStatistics();
    }
    MS_LOG(INFO) << "[ZeroCopy] For Graph " << kernel_graph->ToString() << ", update output "
                 << output_node->DebugString() << " out_idx " << index << " address to "
                 << output_device_addr->GetMutablePtr()
                 << ", shape:" << output_device_addr->kernel_tensor()->GetShapeVector()
                 << ", type: " << TypeIdToString(output_device_addr->type_id())
                 << ", format: " << output_device_addr->format() << ", memory size: " << output_device_addr->GetSize();

    if (output_device_addr->GetPtr() != ge_outputs[idx].GetData() ||
        output_device_addr->GetSize() != ge_outputs[idx].GetSize()) {
      ge_outputs[idx].SetData(reinterpret_cast<uint8_t *>(output_device_addr->GetMutablePtr()),
                              output_device_addr->GetSize(), [](void *) {});
    }
    idx++;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(idx == ge_outputs.size(),
                             "GenerateOutputGeTensor idx not equal to ge_outputs size, idx: " + std::to_string(idx) +
                               ", ge outputs size: " + std::to_string(ge_outputs.size()) +
                               ", kernel graph: " + kernel_graph->ToString());
  return ge_outputs;
}

void GeGraphExecutor::RunInitGraph(const std::string &graph_name) {
  transform::RunOptions run_options;
  run_options.name = "init_subgraph." + graph_name;
  if (transform::GetGraphByName(run_options.name) == nullptr) {
    MS_LOG(INFO) << "Can not find " << run_options.name << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  auto cur_sink_size = CurGraphSinkSize(graph_name);
  if (pre_sink_size_ == cur_sink_size) {
    return;
  }
  pre_sink_size_ = cur_sink_size;
  MS_LOG(INFO) << "Start run init graph: " << run_options.name << ", sink size:" << cur_sink_size;
  std::vector<transform::GeTensorPtr> ge_outputs;
  std::vector<transform::GeTensorPtr> ge_tensors;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    transform::Status ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << run_options.name << " graph failed.";
    }
    MS_LOG(INFO) << "Exec " << run_options.name << " graph success.";
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
