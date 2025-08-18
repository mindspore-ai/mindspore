/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/runtime/actor/actor_common.h"

#include <memory>
#include <unordered_map>
#include <vector>
#include <set>
#include <string>

#include "primitive/framework_op_name.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_op_name.h"
#include "backend/ge_backend/runtime/device_tensor_store.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"
#include "backend/ge_backend/utils/device_address_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
bool ActorDispatcher::is_multi_thread_execution_ = true;
bool ActorDispatcher::is_memory_allocation_sync_ = true;
bool ActorDispatcher::is_memory_free_sync_ = true;

bool IsSuperKernelActor(const AnfNodePtr &node, const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return (kernel_graph->is_graph_run_mode() &&
          ((node == nullptr) || node->isa<CNode>() || kernel_graph->IsChildGraphResult(node)));
}

bool IsRunningFailed(const OpContext<KernelTensor> *context) { return (context->error_info_ != ""); }

bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph,
                        const std::vector<AnfNodePtr> &host_parameters, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(node);

  bool is_parameter_data = node->isa<Parameter>() && (!common::AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()));
  if (!is_parameter_data) {
    return false;
  }
  // Need to be updated every step.
  if (node->has_user_data(kForwardOutput)) {
    return true;
  }

  if (strategy == GraphExecutionStrategy::kStep) {
    MS_EXCEPTION_IF_NULL(graph);
    return graph->execution_order().size() > 1;
  }

  if (graph == nullptr) {
    return true;
  }

  // In control flow, only the parameters of the root funcgraph are in the host data source.
  const auto &front_node = graph->GetFrontAnfByBackendAnf(node);
  bool is_host = ((front_node == nullptr) ||
                  find(host_parameters.begin(), host_parameters.end(), front_node) != host_parameters.end());

  // Judge whether node is internal parameter.
  const auto &internal_front_node = graph->GetFrontNodeByInternalParameter(node);
  if (internal_front_node.first == nullptr && is_host) {
    return true;
  }

  return false;
}

bool IsGraphRootParameter(const AnfNodePtr &node, const KernelGraphPtr &graph,
                          const std::vector<AnfNodePtr> &host_parameters, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(node);
  KernelWithIndex front_node_with_idx{nullptr, 0};
  if (IsInternalParameter(node, graph)) {
    front_node_with_idx = graph->GetFrontNodeByInternalParameter(node);
  } else {
    front_node_with_idx = graph->GetElementInTupleBackendFrontIndexMap(node);
    if (front_node_with_idx.first == nullptr) {
      front_node_with_idx = {AnfAlgo::FetchFrontNodeByBackendNode(node, *graph), 0};
    }
  }
  auto front_node = front_node_with_idx.first;
  MS_EXCEPTION_IF_NULL(front_node);
  bool is_parameter_data = front_node->isa<Parameter>();
  if (is_parameter_data) {
    return true;
  }
  // Need to be updated every step.
  if (node->has_user_data(kForwardOutput)) {
    return true;
  }

  if (strategy == GraphExecutionStrategy::kStep) {
    MS_EXCEPTION_IF_NULL(graph);
    return graph->execution_order().size() > 1;
  }

  if (graph == nullptr) {
    return true;
  }

  // In control flow, only the parameters of the root funcgraph are in the host data source.
  bool is_host = ((front_node == nullptr) ||
                  find(host_parameters.begin(), host_parameters.end(), front_node) != host_parameters.end());

  // Judge whether node is internal parameter.
  const auto &internal_front_node = graph->GetFrontNodeByInternalParameter(node);
  if (internal_front_node.first == nullptr && is_host) {
    return true;
  }

  return false;
}

bool IsSwitchActor(const AnfNodePtr &node) { return common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch); }

bool IsInternalParameter(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (node->isa<Parameter>() && (!common::AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
    //  Judge whether node is internal parameter.
    const auto &front_node = graph->GetOriginFrontNodeByInternalParameter(node);
    if (front_node.first != nullptr) {
      return true;
    }
  }
  return false;
}

bool IsPersistentDeviceTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return true;
  }

  // Maybe the load node, need fetch the real parameter node.
  auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  MS_EXCEPTION_IF_NULL(real_node);
  if (real_node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(real_node->cast<ParameterPtr>())) {
    return true;
  }
  return false;
}

bool IsControlFlowActor(KernelTransformType actor_type) {
  return ((actor_type >= KernelTransformType::kSwitchActor) && (actor_type <= KernelTransformType::kStackActor));
}

bool IsSkippedLaunch(const CNodePtr &kernel, const KernelGraphPtr &kernel_graph) {
  static const char kLaunchSkippedEnv[] = "MS_KERNEL_LAUNCH_SKIP";
  static std::string launch_skipped = common::GetEnv(kLaunchSkippedEnv);
  static bool no_launch_skipped = launch_skipped.empty();
  if (no_launch_skipped) {
    return false;
  }

  static bool launch_skipped_all = (launch_skipped == "all" || launch_skipped == "ALL");
  if (launch_skipped_all) {
    MS_LOG(DEBUG) << "Skip all the launch.";
    return true;
  }

  std::string launch_name = "";
  std::string full_name = "";
  if (kernel != nullptr) {
    launch_name = common::AnfAlgo::GetCNodeName(kernel);
    full_name = kernel->fullname_with_scope();
  } else if (kernel_graph != nullptr) {
    launch_name = kernel_graph->ToString();
    full_name = kernel_graph->ToString();
  } else {
    return false;
  }

  if (launch_skipped == launch_name) {
    MS_LOG(DEBUG) << "Skip the launch of " << full_name;
    return true;
  }

  return false;
}

size_t GetDefragMemoryStepFreq() {
  static size_t defrag_memory_step_freq = 100L;

  static std::once_flag init_flag;
  std::call_once(init_flag, [&]() {
    MS_LOG(INFO) << "Init defrag memory step freq.";
    const auto &value = memory::mem_pool::GetAllocConfigValue(memory::mem_pool::kAllocDefragMemoryStepFreq);
    MS_LOG(INFO) << "Config defrag memory step freq : " << value << ".";
    if (value.size() != 0) {
      std::stringstream sstream(value);
      size_t config_value;
      sstream >> config_value;
      if (config_value != 0) {
        defrag_memory_step_freq = config_value;
      }
    }
    MS_LOG(INFO) << "Defrag memory step freq : " << defrag_memory_step_freq << ".";
  });

  return defrag_memory_step_freq;
}

void UpdateRefCount(const KernelTensorPtr &kernel_tensor, bool is_max_ref_count) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (is_max_ref_count) {
    kernel_tensor->set_original_ref_count(SIZE_MAX);
    MS_LOG(DEBUG) << "Set origin ref count max for kernel tensor:" << kernel_tensor->ToString();
  } else {
    kernel_tensor->IncreaseOriginalRefCount();
    MS_LOG(DEBUG) << "Add origin ref count for kernel tensor:" << kernel_tensor->ToString()
                  << " origin ref count:" << kernel_tensor->original_ref_count();
  }
  kernel_tensor->ResetRefCount();
}

void UpdateRefCount(const AnfNodePtr &node, size_t output_idx, bool is_max_ref_count) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, output_idx, false);
  UpdateRefCount(kernel_tensor, is_max_ref_count);
}

void FreeMemoryByDeviceContext(DeviceTensor *const device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->FreeMemory(device_tensor);
}

void FreeMemoryByValueNode(const std::vector<std::weak_ptr<ValueNode>> &held_by_nodes,
                           const KernelTensorPtr &kernel_tensor) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  device_tensor->ClearHeldByNodes();
  kernel_tensor->set_original_ref_count(SIZE_MAX);
  kernel_tensor->ResetRefCount();

  for (auto &node : held_by_nodes) {
    auto value_node = node.lock();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_device_address(nullptr);
    tensor->set_storage_info(nullptr);
    runtime::DeviceTensorStore::GetInstance().Remove(value_node.get());
  }
}

KernelTransformType FetchKernelTransformType(const AnfNodePtr &node, const KernelGraphPtr &graph,
                                             const std::vector<AnfNodePtr> &host_parameters,
                                             GraphExecutionStrategy strategy) {
  // Fetch kernel graph.
  KernelGraphPtr kernel_graph = nullptr;
  if (graph == nullptr) {
    kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
  } else {
    kernel_graph = graph;
  }
  if (kernel_graph == nullptr) {
    return KernelTransformType::kUnknown;
  }

  // In sink mode, the data exchange between child graphs is expressed as parameters. These parameters are stored
  // in the graph and should be obtained from the super kernel actor.
  if (IsSuperKernelActor(node, kernel_graph)) {
    return KernelTransformType::kSuperKernelActor;
  }

  KernelTransformType type = KernelTransformType::kUnknown;
  MS_EXCEPTION_IF_NULL(node);
  auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  MS_EXCEPTION_IF_NULL(real_node);

  if (IsHostQueueDSActor(real_node, kernel_graph, host_parameters, strategy)) {
    type = KernelTransformType::kHostDataSourceActor;
  } else if (IsInternalParameter(real_node, kernel_graph)) {
    type = KernelTransformType::kInternalParameter;
  } else if (IsPersistentDeviceTensor(real_node)) {
    type = KernelTransformType::kDeviceTensorStore;
  } else {
    // May exist the from kernel that no need link in the pynative mode.
    MS_LOG(DEBUG) << "Invalid from kernel: " << node->DebugString();
  }

  return type;
}

std::string FetchActorName(KernelTransformType kernel_type, const std::string &actor_set_name, const AnfNodePtr &node,
                           const KernelGraphPtr &graph) {
  // Fetch kernel graph.
  KernelGraphPtr kernel_graph = nullptr;
  if (graph == nullptr) {
    kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
  } else {
    kernel_graph = graph;
  }
  if (kernel_graph == nullptr) {
    return "";
  }

  auto real_node = node;
  if (real_node != nullptr) {
    real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  }
  std::string actor_name = "";
  switch (kernel_type) {
    case KernelTransformType::kSuperKernelActor:
      actor_name = kernel_graph->ToString() + kSuperKernelActorNameSuffix;
      break;
    case KernelTransformType::kHostDataSourceActor:
      actor_name = actor_set_name + kHostDSActorNameSuffix;
      break;
    default:
      break;
  }
  return actor_name;
}

std::set<size_t> FetchModifiableRefInputIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Only the auto moand node will modify the input.
  if (!common::AnfAlgo::HasMonadInput(cnode)) {
    return {};
  }

  std::set<size_t> ref_input_indexes;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto &input = cnode->inputs().at(i);
    if (common::AnfAlgo::HasAbstractRef(input)) {
      (void)ref_input_indexes.insert(i - 1);
    }
  }

  return ref_input_indexes;
}

std::set<size_t> FetchModifiableRefOutputIndex(const CNodePtr &cnode, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::set<size_t> ref_output_indexes;

  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < output_num; ++i) {
    session::AnfWithOutIndex output_pair(cnode, i);
    // Only the ref node will modify the ref input corresponding to the output.
    if (!graph->IsInRefOutputMap(output_pair)) {
      continue;
    }
    auto input_pair = graph->GetRefCorrespondOutput(output_pair);
    MS_EXCEPTION_IF_NULL(input_pair.first);
    if (common::AnfAlgo::HasAbstractRef(input_pair.first)) {
      (void)ref_output_indexes.insert(i);
    }
  }
  return ref_output_indexes;
}

std::unordered_map<AnfNode *, std::string> actor_ids;
static size_t actor_index = 0;

std::string GetActorIdByKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (actor_ids.find(node.get()) == actor_ids.end()) {
    MS_LOG(INFO) << "Cannot get actor id by node:" << node->fullname_with_scope();
    return node->fullname_with_scope();
  }
  return actor_ids[node.get()];
}

std::string GenerateActorIdByKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto id = std::to_string(actor_index++) + "_" + node->fullname_with_scope();
  actor_ids[node.get()] = id;
  return id;
}

mindspore::HashMap<size_t, size_t> GetRepeatDeviceAddressIndexPair(const std::vector<KernelTensorPtr> &kernel_tensors) {
  mindspore::HashMap<const void *, std::vector<size_t>> ptr_positions;
  mindspore::HashMap<size_t, size_t> repeat_index;
  for (size_t i = 0; i < kernel_tensors.size(); ++i) {
    if (kernel_tensors[i] != nullptr && kernel_tensors[i]->device_address() != nullptr &&
        kernel_tensors[i]->device_address()->GetPtr() != nullptr) {
      ptr_positions[kernel_tensors[i]->device_address()->GetPtr()].emplace_back(i);
    }
  }
  for (const auto &pair : ptr_positions) {
    if (pair.second.size() <= 1) {
      continue;
    }
    for (size_t i = 1; i < pair.second.size(); ++i) {
      repeat_index[pair.second[i]] = pair.second[0];
    }
  }
  return repeat_index;
}

bool IsInferPhase(const std::string &phase) {
  return phase.find("prefill") != std::string::npos || phase.find("increment") != std::string::npos;
}

size_t FetchInputTensorIndex(const KernelWithIndex &front_node) {
  MS_EXCEPTION_IF_NULL(front_node.first);
  if (common::AnfAlgo::IsDynamicSequence(front_node.first)) {
    return 0;
  }

  const auto &abs = front_node.first->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    return front_node.second;
  }

  return 0;
}

TensorPtr FetchInputTensorByArg(const VectorRef &args, size_t arg_index, const KernelWithIndex &front_node) {
  if (arg_index >= args.size()) {
    MS_LOG(INFO) << "Arg index out of args range, index is " << arg_index << " and args size is " << args.size();
    return nullptr;
  }

  std::vector<tensor::TensorPtr> flatten_tensors;
  AnfAlgo::FlattenInputArg(args[arg_index], front_node.first, &flatten_tensors);
  auto input_tensor_index = FetchInputTensorIndex(front_node);
  if (input_tensor_index >= flatten_tensors.size()) {
    MS_LOG(INFO) << "Input tensor index out of args range, index is " << input_tensor_index << " and tensors size is "
                 << flatten_tensors.size();
    return nullptr;
  }

  auto tensor = flatten_tensors[input_tensor_index];
  // The tensor needs to be converted to contiguous before being given to the actors.
  // After the view feature is supported in the graph mode, the following code will be deleted.
  if (!DeviceAddressUtils::IsContiguousTensor(tensor)) {
    MS_LOG(EXCEPTION) << "The ge backend only support contiguous inputs, please check.";
  }

  return tensor;
}

bool IsEmptySequenceTensor(tensor::Tensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->base_shape_ptr() == nullptr || (!tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    return false;
  }
  const auto &sequence_shape = tensor->base_shape_ptr()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  return sequence_shape->size() == 0;
}

void UpdateDynamicShapeAndSize(tensor::Tensor *input_tensor, KernelTensorPtr kernel_tensor, size_t outer_index,
                               size_t inner_index) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (input_tensor == nullptr || IsEmptySequenceTensor(input_tensor)) {
    return;
  }

  // Update shape.
  const auto &output_kernel_tensor = kernel_tensor;
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  if (input_tensor->base_shape_ptr() == nullptr || (!input_tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    output_kernel_tensor->SetShape(input_tensor->ToAbstract()->GetShape());
    MS_LOG(DEBUG) << "Kernel tensor: " << output_kernel_tensor.get() << ", shape is "
                  << output_kernel_tensor->GetShapeVector();
    return;
  }
  output_kernel_tensor->SetShape(input_tensor->base_shape_ptr());
  MS_LOG(DEBUG) << "Kernel tensor: " << output_kernel_tensor.get() << ", shape is "
                << output_kernel_tensor->GetShapeVector();

  // Update size.
  auto device_format = kernel::GetFormatFromEnumToStr(kernel_tensor->format());
  static const std::set<std::string> kNormalFormat = {
    kOpFormat_DEFAULT, kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN,
  };
  if (kNormalFormat.find(device_format) != kNormalFormat.end()) {
    auto tensor_data_size = input_tensor->DataNBytes();
    MS_LOG(DEBUG) << "Set device address:" << device_tensor << " size from:" << device_tensor->GetSize()
                  << " to:" << tensor_data_size;
    device_tensor->SetSize(tensor_data_size);
  } else {
    MS_LOG(EXCEPTION) << "Can not Update size for 5D format device address";
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
