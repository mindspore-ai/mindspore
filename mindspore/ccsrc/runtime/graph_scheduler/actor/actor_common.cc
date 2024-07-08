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

#include "runtime/graph_scheduler/actor/actor_common.h"
#include <memory>
#include <unordered_map>
#include "ops/framework_op_name.h"
#include "ops/framework_ops.h"
#include "ops/structure_op_name.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "utils/phase.h"
#ifndef BUILD_LITE
#include "runtime/graph_scheduler/actor/kernel_async_launch_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_resize_actor.h"
#endif

namespace mindspore {
namespace runtime {
bool ActorDispatcher::is_multi_thread_execution_ = true;
bool ActorDispatcher::enable_multi_stream_ = false;
bool ActorDispatcher::has_kernel_need_user_data_ = false;
bool ActorDispatcher::is_memory_allocation_sync_ = true;
bool ActorDispatcher::is_memory_free_sync_ = true;
bool ActorDispatcher::enable_runtime_multi_pipeline_ = false;
bool ActorDispatcher::enable_async_launch_kernel_ = false;
bool ActorDispatcher::disable_kbk_sub_graph_execute_ = false;
bool ActorDispatcher::enable_static_shape_ = false;
bool ActorDispatcher::enable_trace_dynamic_memory_ = false;
bool ActorDispatcher::enable_use_trace_memory_ = false;

bool IsRunningFailed(const OpContext<DeviceTensor> *context) { return (context->error_info_ != ""); }

void ComputeThreadNums(size_t *actor_thread_num, size_t *actor_and_kernel_thread_num) {
  MS_EXCEPTION_IF_NULL(actor_thread_num);
  MS_EXCEPTION_IF_NULL(actor_and_kernel_thread_num);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  const size_t cpu_core_num = std::thread::hardware_concurrency();
  auto inter_op_parallel_num = static_cast<size_t>(context_ptr->get_param<uint32_t>(MS_CTX_INTER_OP_PARALLEL_NUM));
  auto runtime_num_threads = static_cast<size_t>(context_ptr->get_param<uint32_t>(MS_CTX_RUNTIME_NUM_THREADS));
  size_t runtime_num_threads_min = std::min(runtime_num_threads, cpu_core_num);
  size_t inter_op_parallel_num_min = std::min(inter_op_parallel_num, cpu_core_num);
  const float kActorUsage = 0.18;
  const size_t kActorThreadMinNum = 1;
  // Compute the actor and kernel thread num.
  // The MemoryManagerActor binds single thread, so if runtime_num_threads is 30, actor num would be 5,
  // kernel num would be 25.
  if (inter_op_parallel_num_min == 0) {
    size_t actor_thread_max_num =
      std::max(static_cast<size_t>(std::floor(runtime_num_threads_min * kActorUsage)), kActorThreadMinNum);
    *actor_thread_num = actor_thread_max_num;
    *actor_and_kernel_thread_num =
      runtime_num_threads_min > *actor_thread_num ? (runtime_num_threads_min) : (*actor_thread_num + 1);
  } else {
    *actor_thread_num = inter_op_parallel_num_min;
    *actor_and_kernel_thread_num = runtime_num_threads_min + *actor_thread_num;
  }

  if (*actor_and_kernel_thread_num > cpu_core_num) {
    MS_LOG(WARNING) << "The total num of thread pool is " << *actor_and_kernel_thread_num
                    << ", but the num of cpu core is " << cpu_core_num
                    << ", please set the threads within reasonable limits.";
  }
}

bool IsDeviceQueueDSActor(const AnfNodePtr &, GraphExecutionStrategy) { return false; }

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

bool IsCustomActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return AnfUtils::IsCustomActorNode(node);
}

bool IsKernelActor(const AnfNodePtr &node, GraphExecutionStrategy) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsCustomActor(node)) {
    return false;
  }

  if (!AnfUtils::IsRealCNodeKernel(node)) {
    return false;
  }

  return true;
}

bool IsSkippedKernelActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsKernelActor(node) && common::AnfAlgo::IsInplaceNode(node, "skip")) {
    return true;
  }
  return false;
}

bool IsRpcActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsKernelActor(node) && (common::AnfAlgo::GetCNodeName(node) == kRpcSendOpName ||
                              common::AnfAlgo::GetCNodeName(node) == kRpcRecvOpName)) {
    return true;
  }
  return false;
}

bool IsInnerControlFlowActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsKernelActor(node) && (common::AnfAlgo::GetCNodeName(node) == "ConditionSwitch" ||
                              common::AnfAlgo::GetCNodeName(node) == "ConditionGather")) {
    return true;
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

bool IsMemoryActor(KernelTransformType actor_type) {
  return ((actor_type == KernelTransformType::kMemoryAllocActor) ||
          (actor_type == KernelTransformType::kMemoryFreeActor));
}

bool IsSkippedLaunch(const CNodePtr &kernel, const KernelGraphPtr &kernel_graph) {
  static std::string launch_skipped = "";
  static bool first_get_launch_skipped_env = true;
  static const char kLaunchSkippedEnv[] = "MS_KERNEL_LAUNCH_SKIP";
  if (first_get_launch_skipped_env) {
    launch_skipped = common::GetEnv(kLaunchSkippedEnv);
    first_get_launch_skipped_env = false;
    if (launch_skipped.empty() && !common::GetEnv(kSimulationLevel).empty()) {
      launch_skipped = "ALL";
    }
  }

  if (launch_skipped.empty()) {
    return false;
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
    MS_LOG(ERROR) << "The luanch kernel or graph is nullptr";
    return false;
  }

  if ((launch_skipped == "ALL") || (launch_skipped == "all") || (launch_skipped == launch_name)) {
    MS_LOG(DEBUG) << "Skip the launch of " << full_name;
    return true;
  }

  return false;
}

bool EnableAsyncInfer() {
  static const char kEnableAsyncInferdEnv[] = "MS_ENABLE_ASYNC_INFER";
  static bool ret = common::GetEnv(kEnableAsyncInferdEnv) == "1";
  return ret;
}

bool EnableTraceMemory() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (!enable_infer_boost) {
    return false;
  }

  if (!EnableKbkSubGraphExecute()) {
    return false;
  }

  static const char kEnableTraceMemoryEnv[] = "MS_ENABLE_TRACE_MEMORY";
  static bool ret = common::GetEnv(kEnableTraceMemoryEnv) == "1";
  if (ret) {
    MS_LOG(INFO) << "Enable trace memory to optimize dynamic memory manage performance.";
  }
  return ret;
}

void ResetTraceMemoryStatus() {
  ActorDispatcher::set_enable_static_shape(false);
  ActorDispatcher::set_enable_trace_dynamic_memory(false);
  ActorDispatcher::set_enable_use_trace_memory(false);
}

bool EnableKbkSubGraphExecute() {
  static const char kEnableKbkSubGraphExecutedEnv[] = "MS_ENABLE_KBK_SUBGRAPH_EXECUTE";
  static bool disable_sub_graph_execute_mode = common::GetEnv(kEnableKbkSubGraphExecutedEnv) == "0";
  if (disable_sub_graph_execute_mode) {
    return false;
  }

  if (ActorDispatcher::disable_kbk_sub_graph_execute()) {
    return false;
  }

  // Only support sub graph execution mode for inference.
  // static const bool enable_internal_kernels = common::GetEnv("MS_ENABLE_INTERNAL_KERNELS") == "on";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_internal_kernels = ms_context->IsEnableInferBoost();
  return enable_internal_kernels;
}

size_t GetDefragMemoryStepFreq() {
  static size_t defrag_memory_step_freq = 100L;

  static std::once_flag init_flag;
  std::call_once(init_flag, [&]() {
    MS_LOG(INFO) << "Init defrag memory step freq.";
    const auto &value = common::GetConfigValue(common::kAllocConf, common::kAllocDefragMemoryStepFreq);
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

bool WaitRuntimePipelineFinish(const OpContext<DeviceTensor> *context, bool wait_kernel_launch_finish) {
#ifndef BUILD_LITE
  if (ActorDispatcher::enable_runtime_multi_pipeline()) {
    KernelAsyncInferActor::GetInstance()->Wait();
    KernelAsyncResizeActor::GetInstance()->Wait();
  }

  if (ActorDispatcher::enable_async_launch_kernel() && wait_kernel_launch_finish) {
    KernelAsyncLaunchActor::GetInstance()->Wait();
  }

  if (ActorDispatcher::enable_async_launch_kernel() && IsRunningFailed(context)) {
    MS_LOG(ERROR) << "Wait runtime pipeline finish and an error occurred: " << context->error_info_;
    return false;
  }
  return true;
#else
  return true;
#endif
}

bool Copy(const DeviceTensor *dst_device_tensor, const DeviceTensor *src_device_tensor) {
  MS_EXCEPTION_IF_NULL(dst_device_tensor);
  MS_EXCEPTION_IF_NULL(src_device_tensor);
  if (src_device_tensor->GetSize() != dst_device_tensor->GetSize()) {
    MS_LOG(INFO) << "Copy size is not equal, input size:" << src_device_tensor->GetSize()
                 << ", output size:" << dst_device_tensor->GetSize();
  }

  // Exist the size alignment in some device, so get the min device size.
  size_t copy_size = std::min(src_device_tensor->GetSize(), dst_device_tensor->GetSize());

  if (dst_device_tensor->GetDeviceType() == src_device_tensor->GetDeviceType()) {
    return dst_device_tensor->SyncDeviceToDevice(src_device_tensor);
  } else if (src_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    // CPU device tensor copy to other device tensor.
    return dst_device_tensor->SyncHostToDevice(copy_size, src_device_tensor->GetPtr());
  } else if (dst_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    // Other device tensor copy to CPU device tensor.
    return src_device_tensor->SyncDeviceToHost(copy_size, dst_device_tensor->GetMutablePtr());
  } else {
    MS_LOG(ERROR) << "Invalid device type, src device type: " << src_device_tensor->GetDeviceType()
                  << ", dst device type: " << dst_device_tensor->GetDeviceType();
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

void FreeMemoryByDeviceContext(DeviceTensor *const device_tensor, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  // The device context may be not accurate in the control flow scene, so need fetch by device name and device id.
  if ((device_context == nullptr) || (device_context->GetDeviceType() != device_tensor->GetDeviceType())) {
    const auto &new_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->device_name(), device_tensor->device_id()});
    MS_EXCEPTION_IF_NULL(new_device_context);
    new_device_context->device_res_manager_->FreeMemory(device_tensor);
  } else {
    device_context->device_res_manager_->FreeMemory(device_tensor);
  }
}

void FreeMemoryByValueNode(const std::vector<std::weak_ptr<ValueNode>> &held_by_nodes, DeviceTensor *device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->ClearHeldByNodes();
  device_tensor->set_original_ref_count(SIZE_MAX);
  device_tensor->ResetRefCount();

  for (auto &node : held_by_nodes) {
    auto value_node = node.lock();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_device_address(nullptr);
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
  if (kernel_graph->is_any_type_input() && node != nullptr && node->isa<CNode>()) {
    return KernelTransformType::kAnyTypeKernelActor;
  }
  // In sink mode, the data exchange between child graphs is expressed as parameters. These parameters are stored
  // in the graph and should be obtained from the super kernel actor.
  if (kernel_graph->is_graph_run_mode() &&
      ((node == nullptr) || node->isa<CNode>() || kernel_graph->IsChildGraphResult(node))) {
    return KernelTransformType::kSuperKernelActor;
  }

  KernelTransformType type = KernelTransformType::kUnknown;
  MS_EXCEPTION_IF_NULL(node);
  auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  MS_EXCEPTION_IF_NULL(real_node);

  if (IsDeviceQueueDSActor(real_node, strategy)) {
    type = KernelTransformType::kDeviceDataSourceActor;
  } else if (IsHostQueueDSActor(real_node, kernel_graph, host_parameters, strategy)) {
    type = KernelTransformType::kHostDataSourceActor;
  } else if (IsCustomActor(real_node)) {
    type = KernelTransformType::kCustomActor;
  } else if (IsKernelActor(real_node, strategy)) {
    type = KernelTransformType::kKernelActor;
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
    case KernelTransformType::kAnyTypeKernelActor:
      actor_name = kernel_graph->ToString() + kAnyTypeKernelActorNameSuffix;
      break;
    case KernelTransformType::kDeviceDataSourceActor:
      actor_name = actor_set_name + kDeviceDSActorNameSuffix + "_" + std::to_string(kernel_graph->graph_id());
      break;
    case KernelTransformType::kHostDataSourceActor:
      actor_name = actor_set_name + kHostDSActorNameSuffix;
      break;
    case KernelTransformType::kCustomActor:
      MS_EXCEPTION_IF_NULL(real_node);
      actor_name = AnfUtils::GetCustomActorName(real_node);
      break;
    case KernelTransformType::kKernelActor:
      MS_EXCEPTION_IF_NULL(real_node);
      actor_name = GetActorIdByKernel(real_node);
      break;
    case KernelTransformType::kKernelInferActor:
      MS_EXCEPTION_IF_NULL(real_node);
      actor_name = kKernelInferActorNamePrefix + real_node->fullname_with_scope();
      break;
    case KernelTransformType::kKernelResizeActor:
      MS_EXCEPTION_IF_NULL(real_node);
      actor_name = kKernelResizeActorNamePrefix + real_node->fullname_with_scope();
      break;
    default:
      break;
  }
  return actor_name;
}

std::set<size_t> FetchModifiableRefInputIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);

  bool has_monad = false;
  std::set<size_t> ref_input_indexes;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto &input = cnode->inputs().at(i);
    if (HasAbstractMonad(input)) {
      has_monad = true;
    }
    if (common::AnfAlgo::HasAbstractRef(input)) {
      (void)ref_input_indexes.insert(i - 1);
    }
  }

  // Only the auto moand node will modify the input.
  if (has_monad) {
    return ref_input_indexes;
  } else {
    return {};
  }
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

bool is_embedding_cache_server() {
  return ps::PSContext::instance()->cache_enable() && ps::PSContext::instance()->is_server();
}

void MemoryTraceManager::ReserveKernelMemoryBlocks(size_t size, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  (*kernel_memory_trace_blocks_)[device_context].reserve(size);
}

void MemoryTraceManager::PickMemoryTrackInfoForGraph(uint32_t graph_id) {
  if (graph_to_kernel_memory_trace_blocks_.find(graph_id) == graph_to_kernel_memory_trace_blocks_.end()) {
    graph_to_kernel_memory_trace_blocks_.emplace(
      graph_id, std::make_shared<std::map<const DeviceContext *, std::vector<KernelMemoryTraceBlockPtr>>>());
  }
  kernel_memory_trace_blocks_ = graph_to_kernel_memory_trace_blocks_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_memory_trace_blocks_);

  if (graph_to_merged_memory_trace_blocks_.find(graph_id) == graph_to_merged_memory_trace_blocks_.end()) {
    graph_to_merged_memory_trace_blocks_.emplace(
      graph_id, std::make_shared<std::map<const DeviceContext *, std::vector<MemoryTraceBlockPtr>>>());
  }
  merged_memory_trace_blocks_ = graph_to_merged_memory_trace_blocks_[graph_id];
  MS_EXCEPTION_IF_NULL(merged_memory_trace_blocks_);

  if (graph_to_kernel_blocks_.find(graph_id) == graph_to_kernel_blocks_.end()) {
    graph_to_kernel_blocks_.emplace(
      graph_id, std::make_shared<mindspore::HashMap<CNodePtr, std::vector<KernelMemoryTraceBlockPtr>>>());
  }
  kernel_to_block_ = graph_to_kernel_blocks_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_to_block_);
}

void MemoryTraceManager::AddKernelMemoryTraceBlock(const KernelMemoryTraceBlockPtr &block,
                                                   const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(block->start_);
  MS_EXCEPTION_IF_NULL(block->end_);
  (*kernel_memory_trace_blocks_)[device_context].emplace_back(block);
}

const std::shared_ptr<std::map<const DeviceContext *, std::vector<MemoryTraceBlockPtr>>>
  &MemoryTraceManager::GetMergeBlocks() {
  return merged_memory_trace_blocks_;
}

const std::shared_ptr<mindspore::HashMap<CNodePtr, std::vector<KernelMemoryTraceBlockPtr>>>
  &MemoryTraceManager::GetAllKernelBlocksnfo() {
  return kernel_to_block_;
}

void MemoryTraceManager::MergeBlocks() {
  merged_memory_trace_blocks_->clear();
  for (auto &item : *kernel_memory_trace_blocks_) {
    auto &device_context = item.first;
    auto &kernel_memory_trace_blocks = item.second;
    MergeBlocksForSameDeviceContext(&kernel_memory_trace_blocks, &((*merged_memory_trace_blocks_)[device_context]));
    MS_LOG(DEBUG) << "The number of merged blocks is " << (*merged_memory_trace_blocks_)[device_context].size()
                  << ", device type: " << device_context->device_context_key().device_name_;
  }
}

void MemoryTraceManager::MergeBlocksForSameDeviceContext(
  std::vector<KernelMemoryTraceBlockPtr> *kernel_memory_trace_blocks,
  std::vector<MemoryTraceBlockPtr> *merged_memory_trace_blocks) {
  MS_EXCEPTION_IF_NULL(kernel_memory_trace_blocks);
  MS_EXCEPTION_IF_NULL(merged_memory_trace_blocks);
  merged_memory_trace_blocks->clear();

  if (kernel_memory_trace_blocks->empty()) {
    MS_LOG(INFO) << "No block to merge.";
    return;
  }

  std::sort(kernel_memory_trace_blocks->begin(), kernel_memory_trace_blocks->end(),
            [](const KernelMemoryTraceBlockPtr &block1, const KernelMemoryTraceBlockPtr &block2) {
              return (block1->start_ < block2->start_) ||
                     ((block1->start_ == block2->start_) && (block1->end_ < block2->end_));
            });
  merged_memory_trace_blocks->emplace_back(std::make_shared<MemoryTraceBlock>((*kernel_memory_trace_blocks)[0]->start_,
                                                                              (*kernel_memory_trace_blocks)[0]->size_));
  (*kernel_memory_trace_blocks)[0]->in_memory_trace_block_index_ = 0;
  for (size_t i = 1; i < kernel_memory_trace_blocks->size(); i++) {
    auto &back = merged_memory_trace_blocks->back();
    auto &block = (*kernel_memory_trace_blocks)[i];
    if (block->start_ >= back->end_) {
      merged_memory_trace_blocks->emplace_back(std::make_shared<MemoryTraceBlock>(block->start_, block->size_));
    } else if (block->end_ > back->end_) {
      back->end_ = block->end_;
      back->size_ = back->end_ - back->start_;
    }
    block->in_memory_trace_block_index_ = merged_memory_trace_blocks->size() - 1;
  }

  // Reset offset
  for (size_t i = 0; i < kernel_memory_trace_blocks->size(); i++) {
    auto &kernel_mem_block = (*kernel_memory_trace_blocks)[i];
    MS_EXCEPTION_IF_NULL(kernel_mem_block);
    const auto &mem_block = (*merged_memory_trace_blocks)[kernel_mem_block->in_memory_trace_block_index_];
    MS_EXCEPTION_IF_NULL(mem_block);
    if (kernel_mem_block->start_ < mem_block->start_) {
      MS_LOG(EXCEPTION) << "Invalid memory block, block start: " << kernel_mem_block->start_
                        << ", block end: " << kernel_mem_block->end_ << ", mem block start: " << mem_block->start_
                        << ", mem block end: " << mem_block->end_;
    }

    kernel_mem_block->offset_in_memory_trace_block_ = kernel_mem_block->start_ - mem_block->start_;
    (*kernel_to_block_)[kernel_mem_block->kernel_].emplace_back(kernel_mem_block);
  }
}

void MemoryTraceManager::Clear() {
  kernel_memory_trace_blocks_->clear();
  merged_memory_trace_blocks_->clear();
  kernel_to_block_->clear();
}

bool IsTwoPhaseInfer() {
  const auto &phase = PhaseManager::GetInstance().phase();
  return phase.find("prefill") != std::string::npos || phase.find("increment") != std::string::npos;
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
}  // namespace runtime
}  // namespace mindspore
