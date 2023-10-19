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
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include "include/backend/debug/profiler/profiling.h"
#include "ge/ge_graph_compile_summary.h"
#include "kernel/kernel_build_info.h"
#include "inc/ops/array_ops.h"
#include "ops/nn_op_name.h"
#include "ops/array_ops.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/compile_cache_context.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const std::set<std::string> kIgnoreGEShapeOps = {kSoftMarginLossOpName};
mindspore::HashMap<std::string, size_t> feature_memorys;
mindspore::HashMap<std::string, size_t> streams;
constexpr size_t kNeedRecycleOutput = 5;

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

transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph, std::map<std::string, ShapeVector> *m_origin_shape) {
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
      m_origin_shape->emplace(para->name(), tensor->shape_c());
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

void RevertOriginShape(const KernelGraphPtr &anf_graph, const std::map<std::string, ShapeVector> &m_origin_shape) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto it = m_origin_shape.find(para->name());
      if (it == m_origin_shape.end()) {
        MS_LOG(ERROR) << "Failed to find input " << para->name() << " in input_shape " << m_origin_shape;
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
    (void)std::transform(ge_shapes.begin(), ge_shapes.end(), std::back_inserter(output_shapes),
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

void SetKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // If kernel build info has been set up. skip
  std::shared_ptr<device::KernelInfo> kernel_info =
    std::dynamic_pointer_cast<device::KernelInfo>(node->kernel_info_ptr());
  kernel::KernelBuildInfoPtr build_info = nullptr;
  if (kernel_info) {
    build_info = kernel_info->GetMutableSelectKernelBuildInfo();
    if (build_info) {
      return;
    }
  }

  if (!kernel_info) {
    kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    node->set_kernel_info(kernel_info);
  }
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

bool BuildFakeGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map) {
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
  if (!AddFakeGraph(anf_graph, init_inputs_map)) {
    MS_LOG(ERROR) << "Add fake graph failed";
    return false;
  }
  GeDeviceResManager::CreateSessionAndGraphRunner();
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return false;
  }
  return true;
}

void ClearForwardOutputAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
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
  if (memo == nullptr) {
    MS_LOG(INFO) << "Start AllocParameterMemory, kernel graph: " << kernel_graph->ToString();
    std::set<KernelGraphPtr> memo_set;
    AllocParameterMemory(kernel_graph, &memo_set);
    MS_LOG(INFO) << "AllocParameterMemory finish.";
    return;
  } else if (memo->find(kernel_graph) != memo->end()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
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
  for (auto &child_graph : kernel_graph->child_graph_order()) {
    AllocParameterMemory(child_graph.lock(), memo);
  }
}

void GeGraphExecutor::BuildInputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_LOG(INFO) << "Start BuildInputDataGeTensor, kernel graph: " << kernel_graph->ToString();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_inputs;
  std::vector<std::pair<AnfNodePtr, size_t>> need_update_input;
  auto input_data_list = kernel_graph->user_data<transform::InputDataList>();
  if (input_data_list == nullptr) {
    MS_LOG(INFO) << "Kernel graph: " << kernel_graph->graph_id() << " input data list is nullptr";
    input_datas_[kernel_graph] = {ge_inputs, need_update_input};
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
      auto abs = cur_inputs.at(cur_inputs_index)->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      while (abs->isa<abstract::AbstractSequence>()) {
        cur_inputs_index++;
        abs = cur_inputs.at(cur_inputs_index)->abstract();
        MS_EXCEPTION_IF_NULL(abs);
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
    MS_LOG(EXCEPTION) << "Not use all cur inputs, cur_inputs_index: " << cur_inputs_index
                      << ", cur_inputs.size(): " << cur_inputs.size() << ", kernel graph: " << kernel_graph->graph_id();
  }
  input_datas_[kernel_graph] = {ge_inputs, need_update_input};
  MS_LOG(INFO) << "BuildInputDataGeTensor finish.";
}

void GeGraphExecutor::BuildOutputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_LOG(INFO) << "Start BuildOutputDataGeTensor, kernel graph: " << kernel_graph->ToString();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_outputs;
  std::vector<std::pair<AnfNodePtr, size_t>> graph_outputs;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    auto index = output_with_index.second;
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
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
  output_datas_[kernel_graph] = {ge_outputs, graph_outputs};
  MS_LOG(INFO) << "BuildOutputDataGeTensor finish.";
}

void GeGraphExecutor::AllocOutputMemory(const KernelGraphPtr &kernel_graph) const {
  MS_LOG(INFO) << "Start AllocOutputMemory, kernel graph: " << kernel_graph->ToString();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  size_t need_alloc_output_cnt = 0;
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    if (output_node->isa<Parameter>()) {
      continue;
    }
    need_alloc_output_cnt++;
  }

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
    bool need_not_alloc = kernel_graph->has_flag(kFlagEnableZeroCopyInGraph) && !output_node->isa<ValueNode>();
    // When ValueNode is a graph output, runtime does not manage this memory
    void *mem = need_not_alloc ? nullptr : ResManager()->AllocateMemory(tensor_size);
    if (common::IsNeedProfileMemory() && !need_not_alloc) {
      MS_LOG(WARNING) << "Need Profile Memory, alloc type: ValueNodeOutput, size:" << tensor_size
                      << ", graph: " << kernel_graph->ToString() << ", node: " << output_node->fullname_with_scope()
                      << ", device address addr: " << mem;
    }
    auto output_device_addr = std::make_shared<AscendDeviceAddress>(mem, tensor_size, kOpFormat_DEFAULT, output_type_id,
                                                                    kAscendDevice, device_id);
    output_device_addr->set_is_ptr_persisted(true);
    if (AscendMemAdapter::GetInstance().IsMemoryPoolRecycle() && need_alloc_output_cnt <= kNeedRecycleOutput) {
      MS_LOG(INFO) << "Set Memory Pool Recycle, graph: " << kernel_graph->ToString()
                   << ", node: " << output_node->fullname_with_scope();
      output_device_addr->set_from_persistent_mem(true);
      output_device_addr->set_need_recycle(true);
    }
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
  MS_LOG(INFO) << "ge graph executor compile graph " << graph->ToString();
  std::map<std::string, ShapeVector> m_origin_shape;
  const auto &tensor_order_map = GetParams(graph, &m_origin_shape);
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto use_compile_cache = compile_cache_context.UseCompileCache();
  if (use_compile_cache) {
    MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";
    if (!BuildFakeGraph(graph, tensor_order_map)) {
      return false;
    }
  } else {
    GEGraphOptimization::GetInstance().OptimizeGEGraph(graph);
    (void)BuildDFGraph(graph, tensor_order_map, false);
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
  if (!graph->is_dynamic_shape()) {
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
  }
  AllocParameterMemory(graph);
  AllocOutputMemory(graph);
  BuildInputDataGeTensor(graph);
  BuildOutputDataGeTensor(graph);
  EnableGraphInputZeroCopy(graph);
  EnableGraphOutputZeroCopy(graph);
  graph->set_run_mode(RunMode::kGraphMode);
  graph->set_memory_managed_by_ge(true);
  if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
    graph->set_is_loop_count_sink(true);
  }
  RevertOriginShape(graph, m_origin_shape);
  return true;
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  MS_EXCEPTION_IF_NULL(graph);
  // cppcheck-suppress unreadVariable
  ContextReset reset_context(device_context_);
  if (IsEnableRefMode()) {
    KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
    return CompileGraph(kg, compile_options);
  } else {
    KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
    MS_EXCEPTION_IF_NULL(kg);
    std::map<std::string, ShapeVector> m_origin_shape;
    const auto &tensor_order_map = GetParams(graph, &m_origin_shape);
    auto &compile_cache_context = CompileCacheContext::GetInstance();
    auto use_compile_cache = compile_cache_context.UseCompileCache();
    if (use_compile_cache) {
      MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";
      if (!BuildFakeGraph(graph, tensor_order_map)) {
        return false;
      }
    } else {
      GEGraphOptimization::GetInstance().OptimizeGEGraph(kg);
      (void)BuildDFGraph(kg, tensor_order_map, false);
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
    RevertOriginShape(kg, m_origin_shape);
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
  return feature_memory_size;
}

bool GeGraphExecutor::RunGraphRefMode(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs) const {
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

  bool is_dynamic_shape = kg->is_dynamic_shape();
  if (is_dynamic_shape) {
    transform::Status ret =
      transform::RegisterExternalAllocator(graph_runner, ResManager()->GetStream(), ResManager()->GetAllocator());
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec graph failed";
    }
  }

  if (AscendMemAdapter::GetInstance().IsMemoryPoolRecycle()) {
    auto max_static_memory_size = ResManager()->GetMaxUsedMemorySize();
    auto iter = feature_memorys.find(graph_name);
    if (iter == feature_memorys.end()) {
      MS_LOG(EXCEPTION) << "Graph " << graph_name << " feature memory not found.";
    }
    auto feature_memory_size = iter->second;
    size_t total_memory_size = max_static_memory_size + feature_memory_size;
    size_t max_hbm_memory_size = static_cast<size_t>(AscendMemAdapter::GetInstance().GetMsUsedHbmSize());
    AscendMemAdapter::GetInstance().UpdateActualPeakMemory(total_memory_size);
    if (total_memory_size > max_hbm_memory_size) {
      MS_LOG(EXCEPTION) << "Memory pool not enough, graph: " << graph_name
                        << ", max_static_memory_size: " << max_static_memory_size
                        << ", feature_memory_size: " << feature_memory_size
                        << ", max_hbm_memory_size: " << max_hbm_memory_size;
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
    auto sync_ret = ResManager()->SyncStream();
    if (!sync_ret) {
      MS_LOG(EXCEPTION) << "Sync stream failed";
    }
    transform::Status ret = transform::UnregisterExternalAllocator(graph_runner, ResManager()->GetStream());
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec graph failed";
    }
    MS_LOG(INFO) << "Run unregister external allocator finish, graph name: " << graph_name;
  }
  // copy output from host to device
  auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  size_t real_output_size = 0;
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    real_output_size++;
    if (is_dynamic_shape) {
      auto real_index = output_node->isa<ValueNode>() ? 0 : idx;
      auto output_addr = AnfAlgo::GetMutableOutputAddr(output_node, real_index, false);
      auto host_type = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
      output_addr->SetSize(ge_outputs[i].GetSize());
      auto actual_shapes = ge_outputs[i].GetTensorDesc().GetShape().GetDims();
      auto &&ge_data_uni = ge_outputs[i].ResetData();
      auto deleter = ge_data_uni.get_deleter();
      auto ge_data = ge_data_uni.release();
      MS_EXCEPTION_IF_NULL(ge_data);
      output_addr->set_is_ptr_persisted(false);
      output_addr->set_from_mem_pool(false);
      output_addr->set_deleter(deleter);
      output_addr->set_ptr(ge_data);
      UpdateOutputNodeShape(output_node, idx, host_type, actual_shapes);
    }
  }
  if (real_output_size != ge_outputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid output size, graph's size " << real_output_size << " tensor size "
                      << ge_outputs.size();
  }

  ClearForwardOutputAddress(kg, device_context_);
  return true;
}

bool GeGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                               std::vector<tensor::Tensor> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  MS_LOG(INFO) << "GE run graph " << graph_name << " start.";
  if (IsEnableRefMode()) {
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
      GilReleaseWithCheck gil_release;
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

  if (!AddDFGraph(anf_graph, init_inputs_map, export_air)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

  if (export_air) {
    // export air can't use session->AddGraph, it will cause atc error.
    return anf_graph;
  }

  GeDeviceResManager::CreateSessionAndGraphRunner();
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
  const auto &input_datas = iter->second.ge_inputs;
  ge_inputs = input_datas;
  bool is_dynamic_shape = kernel_graph->is_dynamic_shape();
  for (auto &kv : iter->second.need_update_input) {
    auto output_addr = AnfAlgo::GetMutableOutputAddr(kv.first, 0, false);
    MS_EXCEPTION_IF_NULL(output_addr);
    auto shapes = trans::GetRuntimePaddingShape(kv.first, 0);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(kv.first, 0);
    auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
    (void)ge_inputs[kv.second].SetTensorDesc(*ge_tensor_desc);
    if (output_addr->GetMutablePtr() == nullptr) {
      // alloc static memory for unused inputs
      // error in ge when set nullptr into ge tensor
      std::vector<size_t> shape = Convert2SizeT(common::AnfAlgo::GetOutputInferShape(kv.first, 0));
      size_t type_size = GetTypeByte(TypeIdToType(common::AnfAlgo::GetOutputInferDataType(kv.first, 0)));
      size_t memory_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>{});
      MS_EXCEPTION_IF_NULL(ResManager());
      auto memory = ResManager()->AllocateMemory(memory_size);
      output_addr->set_ptr(memory);
      output_addr->SetSize(memory_size);
      if (common::IsNeedProfileMemory()) {
        MS_LOG(WARNING) << "Need Profile Memory, alloc type: UnusedInput, size:" << memory_size
                        << ", graph: " << kernel_graph->ToString() << ", node: " << kv.first->fullname_with_scope()
                        << ", device address addr: " << memory;
      }
    }
    if (kv.second >= ge_inputs.size()) {
      MS_LOG(EXCEPTION) << kv.first->DebugString() << ", index: " << kv.second << " is greater than "
                        << ge_inputs.size();
    }
    MS_LOG(DEBUG) << "[ZeroCopy] Update input " << kv.first->DebugString() << " address to "
                  << output_addr->GetMutablePtr();
    size_t memory_size = output_addr->GetSize();
    if (is_dynamic_shape) {
      std::vector<size_t> shape = Convert2SizeT(shapes);
      size_t type_size = GetTypeByte(TypeIdToType(host_type));
      memory_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>{});
    }
    (void)ge_inputs[kv.second].SetData(static_cast<uint8_t *>(output_addr->GetMutablePtr()), memory_size,
                                       [](void *) {});
  }
  for (size_t i = 0; i < ge_inputs.size(); ++i) {
    MS_LOG(INFO) << "Input " << i << " size " << ge_inputs[i].GetSize() << ", graph: " << kernel_graph->graph_id();
  }
  return ge_inputs;
}

std::vector<GeTensor> GeGraphExecutor::GenerateOutputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_outputs;
  auto iter = output_datas_.find(kernel_graph);
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
    auto &output_node = output.first;
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
    MS_LOG(DEBUG) << "[ZeroCopy] Update output " << output_node->DebugString() << " out_idx " << index << " address to "
                  << output_device_addr->GetMutablePtr();
    ge_outputs[idx].SetData(reinterpret_cast<uint8_t *>(output_device_addr->GetMutablePtr()),
                            output_device_addr->GetSize(), [](void *) {});
    idx++;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(idx == ge_outputs.size(),
                             "GenerateOutputGeTensor idx not equal to ge_outputs size, idx: " + std::to_string(idx) +
                               ", ge outputs size: " + std::to_string(ge_outputs.size()) +
                               ", kernel graph: " + kernel_graph->ToString());
  return ge_outputs;
}

void GeGraphExecutor::RunInitGraph(const std::string &graph_name) const {
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
