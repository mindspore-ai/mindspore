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
#include "runtime/dev.h"
#include "runtime/stream.h"
#include "runtime/mem.h"
#include "plugin/device/ascend/hal/hardware/ascend_graph_optimization.h"
#include "include/backend/debug/profiler/profiling.h"
#include "ge/ge_graph_compile_summary.h"
#include "kernel/kernel_build_info.h"
#include "inc/ops/array_ops.h"
#include "ops/array_ops.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const std::set<std::string> kIgnoreGEShapeOps = {kSoftMarginLossOpName};

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

transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph) {
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
      // need ref shape when auto parallel
      auto build_shape = para->abstract()->BuildShape();
      if (build_shape != nullptr) {
        (void)tensor->set_shape(build_shape->cast<abstract::ShapePtr>()->shape());
        MS_LOG(INFO) << "ref abstract Parameter: " << para->name() << ", tensor: " << tensor->ToString();
      }
      res.emplace(para->name(), tensor);
      MS_LOG(DEBUG) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
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
    mindspore::ScopedLongRunning long_running;
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
    MS_LOG(EXCEPTION) << "Invalid output index " << index << ", node " << node->fullname_with_scope() << " has "
                      << total_output_num << " outputs.";
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

  GraphSummary() = default;
  explicit GraphSummary(const ::ge::CompiledGraphSummaryPtr &graph_summary) {
    MS_EXCEPTION_IF_NULL(graph_summary);
    (void)graph_summary->GetConstMemorySize(const_memory_size);
    (void)graph_summary->GetFeatureMemorySize(feature_memory_size);
    (void)graph_summary->GetFeatureMemoryBaseRefreshable(is_feature_memory_refreshable);
    (void)graph_summary->GetStreamNum(stream_num);
    (void)graph_summary->GetEventNum(event_num);
    std::vector<::ge::Shape> ge_shapes;
    (void)graph_summary->GetOutputShapes(ge_shapes);
    std::transform(ge_shapes.begin(), ge_shapes.end(), std::back_inserter(output_shapes),
                   [](const ::ge::Shape &ge_shape) -> ShapeVector { return ge_shape.GetDims(); });
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "const_memory_size[" << const_memory_size << "], feature_memory_size[" << feature_memory_size
       << "], is_feature_memory_refreshable[" << is_feature_memory_refreshable << "], stream_num[" << stream_num
       << "], event_num[" << event_num << "], output size[" << output_shapes.size() << "]";
    if (!output_shapes.empty()) {
      for (size_t i = 0; i < output_shapes.size(); ++i) {
        std::string shape_str = "[";
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
        ss << ", output[" << i << "] shape = " << shape_str;
      }
    }
    return ss.str();
  }
};

std::map<std::string, ParameterPtr> FilterAllParameters(const KernelGraphPtr &kernel_graph) {
  std::map<std::string, ParameterPtr> ret;
  std::vector<AnfNodePtr> todo = kernel_graph->inputs();
  (void)todo.insert(todo.end(), kernel_graph->child_graph_result().begin(), kernel_graph->child_graph_result().end());
  // auto todo = TopoSort(kernel_graph->get_return());
  for (const auto &node : todo) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto parameter = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    std::string name = parameter->name();
    ret.emplace(name, parameter);
  }
  return ret;
}

void SetKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(node);

  std::vector<TypeId> output_infer_types;
  std::vector<std::string> output_formats;
  for (size_t i = 0; i < output_with_indexs.size(); ++i) {
    output_infer_types.emplace_back(
      common::AnfAlgo::GetOutputInferDataType(output_with_indexs[i].first, output_with_indexs[i].second));
    output_formats.emplace_back(kOpFormat_DEFAULT);
  }
  std::shared_ptr<device::KernelInfo> kernel_info =
    std::dynamic_pointer_cast<device::KernelInfo>(node->kernel_info_ptr());
  if (kernel_info == nullptr) {
    kernel_info = std::make_shared<device::KernelInfo>();
    node->set_kernel_info(kernel_info);
  }
  MS_EXCEPTION_IF_NULL(kernel_info);

  kernel::KernelBuildInfoPtr build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (build_info == nullptr) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    build_info = builder->Build();
  }
  MS_EXCEPTION_IF_NULL(build_info);
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
}  // namespace

void GeGraphExecutor::AllocInputHostMemory(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &inputs = kernel_graph->inputs();
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

    auto device_address_ptr =
      std::make_shared<GeHostAddress>(nullptr, tensor_size, kOpFormat_DEFAULT, output_type_id, kAscendDevice, 0);
    device_address_ptr->set_is_ptr_persisted(false);
    AnfAlgo::SetOutputAddr(device_address_ptr, 0, input_node.get());
  }
}

void GeGraphExecutor::AllocOutputHostMemory(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);

    // Parameter's memory is allocated earlier, and there is no need to reallocate memory if Parameter is output.
    if (output_node->isa<Parameter>()) {
      continue;
    }

    auto i = output_with_index.second;
    TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(output_node, i);
    auto output_device_addr =
      std::make_shared<GeHostAddress>(nullptr, 0, kOpFormat_DEFAULT, output_type_id, kAscendDevice, 0);
    AnfAlgo::SetOutputAddr(output_device_addr, i, output_node.get());

    if (common::AnfAlgo::IsNopNode(output_node)) {
      auto [real_node, real_idx] = common::AnfAlgo::GetPrevNodeOutput(output_node, i, true);
      if (real_node != output_node || real_idx != i) {
        AnfAlgo::SetOutputAddr(output_device_addr, real_idx, real_node.get());
      }
    }
  }
}

void GeGraphExecutor::AllocConstMemory(const transform::RunOptions &options, size_t memory_size) const {
  if (memory_size == 0) {
    return;
  }
  auto memory = ResManager()->AllocateMemory(memory_size);
  auto graph_runner = transform::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  auto ret = graph_runner->SetConstMemory(options, memory, memory_size);
  if (ret != transform::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "SetConstMemory for graph " << options.name << " failed.";
  }
}

void GeGraphExecutor::AllocFeatureMemory(const transform::RunOptions &options, size_t memory_size) const {
  if (memory_size == 0) {
    return;
  }
  auto memory_manager = ResManager()->mem_manager_;
  MS_EXCEPTION_IF_NULL(memory_manager);
  memory_manager->ResetDynamicMemory();
  auto memory = memory_manager->MallocWorkSpaceMem(memory_size);
  auto graph_runner = transform::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  auto ret = graph_runner->UpdateFeatureMemory(options, memory, memory_size);
  if (ret != transform::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "UpdateFeatureMemory for graph " << options.name << " failed.";
  }
}

void GeGraphExecutor::AllocParameterMemory(const KernelGraphPtr &kernel_graph, std::set<KernelGraphPtr> *memo) const {
  // Set Device Type to be same as Host Type, AssignStaticMemoryInput will ignore parameters without DeviceType
  if (memo == nullptr) {
    MS_LOG(INFO) << "Start AllocParameterMemory.";
    std::set<KernelGraphPtr> memo_set;
    AllocParameterMemory(kernel_graph, &memo_set);
    MS_LOG(INFO) << "AllocParameterMemory finish.";
    return;
  } else if (memo->find(kernel_graph) != memo->end()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  memo->insert(kernel_graph);
  auto parameters = FilterAllParameters(kernel_graph);
  for (auto iter : parameters) {
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
  auto runtime_instance = dynamic_cast<AscendKernelRuntime *>(
    device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id));
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(*kernel_graph.get());
  for (auto &child_graph : kernel_graph->child_graph_order()) {
    AllocParameterMemory(child_graph.lock(), memo);
  }
}

void GeGraphExecutor::BuildInputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_LOG(INFO) << "Start BuildInputDataGeTensor.";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_inputs;
  auto input_data_list = kernel_graph->user_data<transform::InputDataList>();
  if (input_data_list == nullptr) {
    MS_LOG(INFO) << "Kernel graph: " << kernel_graph->graph_id() << " input data list is nullptr";
    input_datas_[kernel_graph] = ge_inputs;
    return;
  }
  auto parameters = FilterAllParameters(kernel_graph);
  using Data = ::ge::op::Data;
  using RefData = ::ge::op::RefData;
  const auto &cur_inputs = kernel_graph->get_inputs();
  size_t cur_inputs_index = 0;
  for (const auto &op : input_data_list->input_datas) {
    AnfNodePtr node = nullptr;
    auto name = op->GetName();
    if (auto data = std::dynamic_pointer_cast<Data>(op); data != nullptr) {
      while (HasAbstractMonad(cur_inputs.at(cur_inputs_index))) {
        cur_inputs_index++;
      }
      node = cur_inputs.at(cur_inputs_index);
      cur_inputs_index++;
    } else if (auto ref_data = std::dynamic_pointer_cast<RefData>(op); ref_data != nullptr) {
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
    } else {
      MS_LOG(EXCEPTION) << "Op " << name << " is invalid type " << op->GetOpType() << " as graph input.";
    }

    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(INFO) << "Build input ge tensor: " << name << ", kernel graph: " << kernel_graph->graph_id();
    auto output_addr = AnfAlgo::GetMutableOutputAddr(node, 0, false);
    auto shapes = trans::GetRuntimePaddingShape(node, 0);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
    auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    GeTensor ge_tensor(*ge_tensor_desc);
    if (output_addr->GetMutablePtr() != nullptr) {
      ge_tensor.SetData(reinterpret_cast<uint8_t *>(output_addr->GetMutablePtr()), output_addr->GetSize(),
                        [](void *) {});
      MS_LOG(INFO) << "ge input data " << ge_inputs.size() << " name: " << name << " size: " << output_addr->GetSize();
    } else {
      input_datas_index_[kernel_graph].emplace(node, ge_inputs.size());
    }
    ge_inputs.emplace_back(std::move(ge_tensor));
  }
  while (cur_inputs_index < cur_inputs.size() && HasAbstractMonad(cur_inputs.at(cur_inputs_index))) {
    cur_inputs_index++;
  }
  if (cur_inputs_index != cur_inputs.size()) {
    MS_LOG(EXCEPTION) << "Not use all cur inputs, cur_inputs_index: " << cur_inputs_index
                      << ", cur_inputs.size(): " << cur_inputs.size() << ", kernel graph: " << kernel_graph->graph_id();
  }
  input_datas_[kernel_graph] = ge_inputs;
  MS_LOG(INFO) << "BuildInputDataGeTensor finish.";
}

void GeGraphExecutor::AllocOutputMemory(const KernelGraphPtr &kernel_graph,
                                        const std::vector<ShapeVector> &outputs_shape) const {
  MS_LOG(INFO) << "Start AllocOutputMemory.";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    auto index = output_with_index.second;
    MS_EXCEPTION_IF_NULL(output_node);
    SetKernelInfo(output_node);

    // Parameter's memory is allocated earlier, and there is no need to reallocate memory if Parameter is output.
    if (output_node->isa<Parameter>()) {
      continue;
    }

    auto real_index = output_node->isa<ValueNode>() ? 0 : index;
    TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
    size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
    auto shapes = trans::GetRuntimePaddingShape(output_node, real_index);
    auto tensor_size =
      shapes.empty() ? type_size : std::accumulate(shapes.begin(), shapes.end(), type_size, std::multiplies<size_t>());
    // When ValueNode is a graph output, runtime does not manage this memory
    void *mem = kernel_graph->has_flag(kFlagEnableZeroCopyInGraph) && !output_node->isa<ValueNode>()
                  ? nullptr
                  : ResManager()->AllocateMemory(tensor_size);
    auto output_device_addr = std::make_shared<AscendDeviceAddress>(mem, tensor_size, kOpFormat_DEFAULT, output_type_id,
                                                                    kAscendDevice, device_id);
    output_device_addr->set_is_ptr_persisted(true);
    AnfAlgo::SetOutputAddr(output_device_addr, index, output_node.get());

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

bool GeGraphExecutor::CompileGraph(const KernelGraphPtr &graph,
                                   const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  AscendGraphOptimization::GetInstance().OptimizeGEGraph(graph);
  (void)BuildDFGraph(graph, GetParams(graph), false);
  SetDynamicShapeAttr(graph);
  transform::RunOptions run_options;
  run_options.name = GetGraphName(graph);
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  // create loop var
  RunInitGraph(run_options.name);
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
  AllocConstMemory(run_options, summary.const_memory_size);
  AllocFeatureMemory(run_options, summary.feature_memory_size);
  AllocParameterMemory(graph);
  BuildInputDataGeTensor(graph);
  AllocOutputMemory(graph, summary.output_shapes);
  EnableGraphInputZeroCopy(graph);
  EnableGraphOutputZeroCopy(graph);
  graph->set_run_mode(RunMode::kGraphMode);
  graph->set_memory_managed_by_ge(true);
  if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
    graph->set_is_loop_count_sink(true);
  }
  return true;
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  MS_EXCEPTION_IF_NULL(graph);
  if (common::IsEnableRefMode()) {
    KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
    return CompileGraph(kg, compile_options);
  } else {
    MS_EXCEPTION_IF_NULL(graph);
    KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
    MS_EXCEPTION_IF_NULL(kg);
    AscendGraphOptimization::GetInstance().OptimizeGEGraph(kg);
    (void)BuildDFGraph(kg, GetParams(kg), false);
    SetDynamicShapeAttr(kg);
    AllocInputHostMemory(kg);
    AllocOutputHostMemory(kg);
    kg->set_run_mode(RunMode::kGraphMode);
    if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
      kg->set_is_loop_count_sink(true);
    }
    // copy init weight to device
    RunGEInitGraph(kg);
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

bool GeGraphExecutor::RunGraphRefMode(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  MS_LOG(INFO) << "GE run graph " << graph_name << " start.";
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
  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    MS_LOG(INFO) << "Run graph begin, inputs size is: " << inputs.size() << ", " << graph_name;
    transform::Status ret =
      transform::RunGraphWithStreamAsync(graph_runner, run_options, ResManager()->GetStream(), ge_inputs, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec graph failed";
    }
    MS_LOG(INFO) << "Run graph finish, outputs size is: " << ge_outputs.size() << ", " << graph_name;
  }

  // copy output from host to device
  auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  size_t real_output_size = 0;
  for (const auto &output : graph_outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    MS_EXCEPTION_IF_NULL(output_with_index.first);
    if (HasAbstractMonad(output_with_index.first)) {
      continue;
    }
    real_output_size++;
  }
  if (real_output_size != ge_outputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid output size, graph's size " << real_output_size << " tensor size "
                      << ge_outputs.size();
  }
  return true;
}

bool GeGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                               std::vector<tensor::Tensor> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  MS_LOG(INFO) << "GE run graph " << graph_name << " start.";
  if (common::IsEnableRefMode()) {
    if (!RunGraphRefMode(graph, inputs)) {
      return false;
    }
  } else {
    profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, 1, 0, 0);
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
      mindspore::ScopedLongRunning long_running;
      MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size();
      transform::Status ret = transform::RunGraphAsync(graph_runner, run_options, ge_inputs, &ge_outputs);
      MS_LOG(DEBUG) << "Run graph finish, outputs size is: " << ge_outputs.size();
      if (ret == transform::Status::NOT_FOUND) {
        MS_LOG(WARNING) << "The Graph[" << graph_name << "] is not found, skip run it.";
        profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, 1, 0, 1);
        return true;
      } else if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec graph failed";
      }
    }
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
  if (graph->has_flag(transform::kGraphFlagHasGetNext)) {
    MS_LOG(DEBUG) << "Reset ConfigManager, graph: " << graph_name;
    ConfigManager::GetInstance().ResetConfig();
    ConfigManager::GetInstance().ResetIterNum();
  }
  profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, 1, 0, 1);
  MS_LOG(DEBUG) << "GE run graph end.";
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
      draw::Draw("anf_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph.ir", anf_graph, true);
  }
#endif
  (void)setenv("GE_TRAIN", IsGeTrain() ? "1" : "0", 1);
  if (!AddDFGraph(anf_graph, init_inputs_map, export_air)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

  GeDeviceResManager::CreateSessionAndGraphRunner(IsGeTrain());
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return nullptr;
  }

  return anf_graph;
}

std::vector<GeTensor> GeGraphExecutor::GenerateInputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_inputs;
  auto iter = input_datas_.find(kernel_graph);
  if (iter == input_datas_.end()) {
    return ge_inputs;
  }
  const auto &input_datas = iter->second;
  ge_inputs = input_datas;
  auto input_index_iter = input_datas_index_.find(kernel_graph);
  if (input_index_iter != input_datas_index_.end()) {
    for (auto &kv : input_index_iter->second) {
      auto output_addr = AnfAlgo::GetMutableOutputAddr(kv.first, 0, false);
      MS_EXCEPTION_IF_NULL(output_addr);
      if (output_addr->GetMutablePtr() == nullptr) {
        MS_LOG(EXCEPTION) << "Input " << kv.first->fullname_with_scope()
                          << " address is nullptr, kernel graph: " << kernel_graph->ToString() << ", "
                          << kv.first->DebugString();
      }
      if (kv.second >= ge_inputs.size()) {
        MS_LOG(EXCEPTION) << kv.first->DebugString() << ", index: " << kv.second << " is greater than "
                          << ge_inputs.size();
      }
      ge_inputs[kv.second].SetData(reinterpret_cast<uint8_t *>(output_addr->GetMutablePtr()), output_addr->GetSize(),
                                   [](void *) {});
    }
  }
  for (size_t i = 0; i < ge_inputs.size(); ++i) {
    MS_LOG(INFO) << "Input " << i << " size " << ge_inputs[i].GetSize() << ", graph: " << kernel_graph->graph_id();
  }
  return ge_inputs;
}

std::vector<GeTensor> GeGraphExecutor::GenerateOutputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_outputs;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    auto index = output_with_index.second;
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    auto output_device_addr = AnfAlgo::GetMutableOutputAddr(output_node, index, false);
    auto real_index = output_node->isa<ValueNode>() ? 0 : index;
    auto shapes = trans::GetRuntimePaddingShape(output_node, real_index);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
    auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    GeTensor ge_tensor(*ge_tensor_desc);
    MS_LOG(INFO) << "Output addr " << output_device_addr->GetMutablePtr();
    if (output_device_addr->GetMutablePtr() == nullptr) {
      MS_LOG(EXCEPTION) << "Input " << output_node->fullname_with_scope() << ", index: " << index
                        << " address is nullptr, kernel graph: " << kernel_graph->ToString();
    }
    ge_tensor.SetData(reinterpret_cast<uint8_t *>(output_device_addr->GetMutablePtr()), output_device_addr->GetSize(),
                      [](void *) {});
    ge_outputs.emplace_back(std::move(ge_tensor));
  }
  return ge_outputs;
}

void GeGraphExecutor::RunInitGraph(const std::string &graph_name) {
  transform::RunOptions run_options;
  run_options.name = "init_subgraph." + graph_name;
  if (transform::GetGraphByName(run_options.name) == nullptr) {
    MS_LOG(WARNING) << "Can not find " << run_options.name
                    << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

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
