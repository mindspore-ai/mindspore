/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/device_tensor_store.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace runtime {
bool ActorDispatcher::is_multi_thread_execution_ = true;

void ComputeThreadNums(size_t *actor_thread_num, size_t *actor_and_kernel_thread_num) {
  MS_EXCEPTION_IF_NULL(actor_thread_num);
  MS_EXCEPTION_IF_NULL(actor_and_kernel_thread_num);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  const size_t cpu_core_num = std::thread::hardware_concurrency() - 1;
  auto runtime_num_threads = static_cast<size_t>(context_ptr->get_param<uint32_t>(MS_CTX_RUNTIME_NUM_THREADS));
  size_t runtime_num_threads_min = std::min(runtime_num_threads, cpu_core_num);
  const float kActorUsage = 0.2;
  const size_t kActorThreadMinNum = 2;
  size_t actor_thread_max_num =
    std::max(static_cast<size_t>(std::floor(runtime_num_threads_min * kActorUsage)), kActorThreadMinNum);
  // Compute the actor thread num.
  // The MemoryManagerActor binds single thread, and the other actors share one thread at least, so the min num is 2.
  *actor_thread_num = runtime_num_threads_min < kActorThreadMinNum ? kActorThreadMinNum : runtime_num_threads_min;
  *actor_thread_num = *actor_thread_num > actor_thread_max_num ? actor_thread_max_num : *actor_thread_num;

  // Compute the actor and kernel thread num.
  *actor_and_kernel_thread_num =
    runtime_num_threads_min > *actor_thread_num ? runtime_num_threads_min : (*actor_thread_num + 1);
  if (runtime_num_threads != *actor_and_kernel_thread_num) {
    MS_LOG(WARNING) << "The runtime_num_threads is " << runtime_num_threads
                    << ", but actually the num of threads in threadpool is " << *actor_and_kernel_thread_num;
  }
}

bool IsDeviceQueueDSActor(const AnfNodePtr &node, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(node);
  if (strategy == GraphExecutionStrategy::kStep) {
    return false;
  }

  if (node->isa<CNode>() && (AnfAlgo::GetCNodeName(node) == kGetNextOpName)) {
    return true;
  }
  return false;
}

bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph,
                        const std::vector<AnfNodePtr> &host_parameters, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(node);

  bool is_parameter_data = node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()));
  if (!is_parameter_data) {
    return false;
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
  bool is_host = ((front_node == nullptr) || host_parameters.empty() ||
                  find(host_parameters.begin(), host_parameters.end(), front_node) != host_parameters.end());

  // Judge whether node is internal parameter.
  const auto &internal_front_node = graph->GetFrontNodeByInternalParameter(node);
  if (internal_front_node.first == nullptr && is_host) {
    return true;
  }

  return false;
}

bool IsSwitchActor(const AnfNodePtr &node) { return AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch); }

bool IsInternalParameter(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
    //  Judge whether node is internal parameter.
    const auto &front_node = graph->GetFrontNodeByInternalParameter(node);
    if (front_node.first != nullptr) {
      return true;
    }
  }
  return false;
}

bool IsKernelActor(const AnfNodePtr &node, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealCNodeKernel(node)) {
    return false;
  }

  if (strategy == GraphExecutionStrategy::kStep) {
    return true;
  }

  return (AnfAlgo::GetCNodeName(node) != kGetNextOpName);
}

bool IsSkippedKernelActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsKernelActor(node) && AnfAlgo::IsInplaceNode(node, "skip")) {
    return true;
  }
  return false;
}

bool IsPersistentDeviceTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return true;
  }
  if (node->isa<Parameter>() && AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>())) {
    return true;
  }
  return false;
}

bool Copy(const DeviceTensor *dst_device_tensor, const DeviceTensor *src_device_tensor) {
  MS_EXCEPTION_IF_NULL(dst_device_tensor);
  MS_EXCEPTION_IF_NULL(src_device_tensor);
  if (src_device_tensor->GetSize() != dst_device_tensor->GetSize()) {
    MS_LOG(WARNING) << "Copy size is not equal, input size:" << src_device_tensor->GetSize()
                    << ", output size:" << dst_device_tensor->GetSize();
  }

  // Exist the size alignment in some device, so get the min device size.
  size_t copy_size = std::min(src_device_tensor->GetSize(), dst_device_tensor->GetSize());

  if (src_device_tensor->DeviceType() == device::DeviceAddressType::kCPU) {
    // CPU device tensor copy to other device tensor.
    return dst_device_tensor->SyncHostToDevice(copy_size, src_device_tensor->GetPtr());
  } else if (dst_device_tensor->DeviceType() == device::DeviceAddressType::kCPU) {
    // Other device tensor copy to CPU device tensor.
    return src_device_tensor->SyncDeviceToHost(copy_size, dst_device_tensor->GetMutablePtr());
  } else if (dst_device_tensor->DeviceType() == src_device_tensor->DeviceType()) {
    return dst_device_tensor->SyncDeviceToDevice(src_device_tensor);
  } else {
    MS_LOG(ERROR) << "Invalid device type, src device type: " << src_device_tensor->DeviceType()
                  << ", dst device type: " << dst_device_tensor->DeviceType();
    return false;
  }
}

void UpdateRefCount(DeviceTensor *const device_tensor, bool is_max_ref_count) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (is_max_ref_count) {
    device_tensor->set_original_ref_count(SIZE_MAX);
  } else {
    device_tensor->IncreaseOriginalRefCount();
  }
  device_tensor->ResetRefCount();
}

void UpdateRefCount(const AnfNodePtr &node, size_t output_idx, bool is_max_ref_count) {
  MS_EXCEPTION_IF_NULL(node);
  auto device_tensor = AnfAlgo::GetMutableOutputAddr(node, output_idx, false);
  UpdateRefCount(device_tensor.get(), is_max_ref_count);
}

AnfNodePtr FetchFrontNodeByBackendNode(const AnfNodePtr &backend_node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(graph);
  auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(backend_node, *graph);
  return front_node;
}

KernelWithIndex FetchFrontNodeWithIndexByGraphOutput(const KernelWithIndex &output_with_index,
                                                     const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto front_node_with_index = graph->GetFrontNodeWithIndexByGraphOutput(output_with_index);
  // PyNative forward graph does not has front node, using backend node instead.
  if (front_node_with_index.first == nullptr) {
    front_node_with_index = output_with_index;
  }
  return front_node_with_index;
}

KernelGraphPtr FetchKernelGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &func_graph = node->func_graph();
  if (func_graph == nullptr) {
    return nullptr;
  } else {
    return func_graph->cast<KernelGraphPtr>();
  }
}

KernelTransformType FetchKernelTransformType(const AnfNodePtr &node, const KernelGraphPtr &graph,
                                             const std::vector<AnfNodePtr> &host_parameters,
                                             GraphExecutionStrategy strategy) {
  // Fetch kernel graph.
  KernelGraphPtr kernel_graph = nullptr;
  if (graph == nullptr) {
    kernel_graph = FetchKernelGraph(node);
  } else {
    kernel_graph = graph;
  }
  if (kernel_graph == nullptr) {
    return KernelTransformType::kUnknown;
  }

  // In sink mode, the data exchange between child graphs is expressed as parameters. These parameters are stored
  // in the graph and should be obtained from the super kernel actor.
  if (kernel_graph->is_executing_sink() &&
      ((node == nullptr) || node->isa<CNode>() || kernel_graph->IsChildGraphResult(node))) {
    return KernelTransformType::kSuperKernelActor;
  }

  KernelTransformType type = KernelTransformType::kUnknown;
  MS_EXCEPTION_IF_NULL(node);
  if (IsDeviceQueueDSActor(node, strategy)) {
    type = KernelTransformType::kDeviceDataSourceActor;
  } else if (IsHostQueueDSActor(node, kernel_graph, host_parameters, strategy)) {
    type = KernelTransformType::kHostDataSourceActor;
  } else if (IsKernelActor(node, strategy)) {
    type = KernelTransformType::kKernelActor;
  } else if (IsInternalParameter(node, kernel_graph)) {
    type = KernelTransformType::kInternalParameter;
  } else if (IsPersistentDeviceTensor(node)) {
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
    kernel_graph = FetchKernelGraph(node);
  } else {
    kernel_graph = graph;
  }
  if (kernel_graph == nullptr) {
    return "";
  }

  std::string actor_name = "";
  switch (kernel_type) {
    case KernelTransformType::kSuperKernelActor:
      actor_name = kernel_graph->ToString() + "_SuperKernelActor";
      break;
    case KernelTransformType::kDeviceDataSourceActor:
      actor_name = actor_set_name + "_DeviceDSActor" + "_" + std::to_string(kernel_graph->graph_id());
      break;
    case KernelTransformType::kHostDataSourceActor:
      actor_name = actor_set_name + "_HostDSActor";
      break;
    case KernelTransformType::kKernelActor:
      MS_EXCEPTION_IF_NULL(node);
      actor_name = node->fullname_with_scope();
      break;
    default:
      break;
  }
  return actor_name;
}

bool HasAbstractRef(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto &abs = node->abstract();
  return (abs != nullptr) && abs->isa<abstract::AbstractRef>();
}

std::set<size_t> FetchModifiableRefInputIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Only the auto moand node will modify the input.
  if (!HasAbstractMonad(cnode)) {
    return {};
  }

  std::set<size_t> ref_input_indexes;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto &input = cnode->inputs().at(i);
    if (HasAbstractRef(input)) {
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
    if (HasAbstractRef(input_pair.first)) {
      (void)ref_output_indexes.insert(i);
    }
  }
  return ref_output_indexes;
}
}  // namespace runtime
}  // namespace mindspore
