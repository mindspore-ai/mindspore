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

#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include <memory>
#include <map>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include "ir/tensor_new.h"
#include "primitive/framework_op_name.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_op_name.h"
#include "include/runtime/hardware_abstract/kernel_base/device_tensor_store.h"
#include "include/utils/callback.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/utils/anfalgo.h"
#include "include/cluster/topology/ps_context.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "backend/ms_backend/runtime/actors/base/kernel_async_launch_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_infer_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_resize_actor.h"
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "include/backend/common/device_address_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "utils/ms_exception.h"
#include "include/runtime/pipeline/pipeline.h"

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
bool ActorDispatcher::enable_sub_graph_execute_for_cur_actor_set_ = false;
bool ActorDispatcher::enable_static_shape_ = false;
bool ActorDispatcher::enable_trace_dynamic_memory_ = false;
bool ActorDispatcher::enable_use_trace_memory_ = false;
bool ActorDispatcher::enable_input_optimize_for_cur_actor_set_ = true;
bool ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_actor_set_ = false;
bool ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_step_ = false;

bool IsSuperKernelActor(const AnfNodePtr &node, const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return (kernel_graph->is_graph_run_mode() &&
          ((node == nullptr) || node->isa<CNode>() || kernel_graph->IsChildGraphResult(node)));
}

bool IsRunningFailed(const OpContext<KernelTensor> *context) {
  static auto task_fail_cb = GET_COMMON_CALLBACK(TaskFailCallback, void, std::string *);
  if (task_fail_cb != nullptr) {
    task_fail_cb(&const_cast<OpContext<KernelTensor> *>(context)->error_info_);
  }

  return (context->error_info_ != "");
}

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

  if (!node->isa<Parameter>()) {
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

bool IsKernelActor(const AnfNodePtr &node, GraphExecutionStrategy) {
  MS_EXCEPTION_IF_NULL(node);
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

bool EnableTraceMemory() {
  static const bool enable_mem_tracker = memory::mem_pool::IsEnableMemTrack();
  if (enable_mem_tracker) {
    return false;
  }

  // capture graph not support trace memory.
  if (EnableCaptureGraph()) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (!enable_infer_boost) {
    return false;
  }

  if (!EnableKbkSubGraphExecute()) {
    return false;
  }

  static const char kEnableTraceMemoryEnv[] = "MS_ENABLE_TRACE_MEMORY";
  static bool disable_trace_memory = common::GetEnv(kEnableTraceMemoryEnv) == "off";
  if (disable_trace_memory) {
    return false;
  }

  MS_LOG(INFO) << "Enable trace memory to optimize dynamic memory manage performance.";
  return true;
}

void ResetTraceMemoryStatus() {
  ActorDispatcher::set_enable_static_shape(false);
  ActorDispatcher::set_enable_trace_dynamic_memory(false);
  ActorDispatcher::set_enable_use_trace_memory(false);

  ActorDispatcher::set_enable_parallel_dispatch_kernel_for_cur_actor_set(false);
  ActorDispatcher::set_enable_parallel_dispatch_kernel_for_cur_step(false);
}

void ResetPipelineStatus() {
  ActorDispatcher::set_enable_async_launch_kernel(false);
  ActorDispatcher::set_enable_runtime_multi_pipeline(false);
}

void ResetPipelineAndTraceMemoryStatus() {
  ResetPipelineStatus();
  ResetTraceMemoryStatus();
}

bool EnableKbkSubGraphExecute() {
  static bool disable_sub_graph_mode = runtime::IsDisableRuntimeConfig(runtime::kRuntimeKbkSubGraphMode);
  if (disable_sub_graph_mode) {
    return false;
  }

  if (ActorDispatcher::disable_kbk_sub_graph_execute()) {
    return false;
  }

  if (!EnableRuntimePipeline()) {
    return false;
  }

  if (!ActorDispatcher::enable_sub_graph_execute_for_cur_actor_set()) {
    MS_LOG(DEBUG) << "Disable sub graph execute for current graph.";
    return false;
  }

  return true;
}

bool EnableInputOptimize() {
  static bool disable_input_optimize = runtime::IsDisableRuntimeConfig(runtime::kRuntimeInputOptimize);
  if (disable_input_optimize) {
    return false;
  }

  if (!EnableKbkSubGraphExecute()) {
    return false;
  }

  if (!ActorDispatcher::enable_input_optimize_for_cur_actor_set()) {
    return false;
  }

  return true;
}

bool EnableRuntimePipeline() {
  static bool disable_runtime_pipeline = runtime::IsDisableRuntimeConfig(runtime::kRuntimePipeline);
  if (disable_runtime_pipeline) {
    return false;
  }
  return true;
}

bool EnableParallelDispatchKernel() {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  static bool enable_parallel_dispatch_kernel = runtime_conf_instance->IsKernelLaunchGroupConfigured();
  return enable_parallel_dispatch_kernel;
}

bool EnableCaptureGraph() {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  bool enable_capture_graph = runtime_conf_instance->GetEnableKernelLaunchCapture();
  return enable_capture_graph;
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

bool WaitRuntimePipelineFinish(const OpContext<KernelTensor> *context, const std::string &name,
                               bool wait_kernel_launch_finish) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  if (ActorDispatcher::enable_runtime_multi_pipeline()) {
    if (EnableRuntimeNewPipeline()) {
      RuntimePipeline::GetInstance().infer_queue()->Wait();
      RuntimePipeline::GetInstance().resize_queue()->Wait();
    } else {
      KernelAsyncInferActor::GetInstance()->Wait();
      KernelAsyncResizeActor::GetInstance()->Wait();
    }
  }

  if (ActorDispatcher::enable_async_launch_kernel() && wait_kernel_launch_finish) {
    if (EnableRuntimeNewPipeline()) {
      RuntimePipeline::GetInstance().launch_queue()->Wait();
    } else {
      KernelAsyncLaunchActor::GetInstance()->Wait();
    }
  }
  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kWaitTaskFinish, name, false);

  if (ActorDispatcher::enable_async_launch_kernel() && IsRunningFailed(context)) {
    MS_LOG(INFO) << "Wait runtime pipeline finish and an error occurred: " << context->error_info_;
    return false;
  }
  return true;
}

bool SyncAllStreamForDeviceAddress(const DeviceTensorPtr &dst_device_tensor, const DeviceTensorPtr &src_device_tensor,
                                   uint32_t stream_id, bool sync_stream_on_demand) {
  if (dst_device_tensor == nullptr || src_device_tensor == nullptr) {
    MS_LOG(EXCEPTION) << "Invalidate device tensor, dst_device_tensor : " << dst_device_tensor
                      << ", src_device_tensor : " << src_device_tensor;
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Sync all stream for device address, stream id : " << stream_id
    << ", dst : " << dst_device_tensor->GetDeviceType() << ", stream id : " << dst_device_tensor->stream_id()
    << ", src : " << src_device_tensor->GetDeviceType() << ", stream id : " << src_device_tensor->stream_id();

  static bool enable_sync_stream_on_demand = []() -> bool {
    auto ret = runtime::IsEnableRuntimeConfig(runtime::kRuntimeSyncStreamOnDemand);
    MS_LOG(INFO) << "Runtime config, sync stream on demand : "
                 << runtime::GetRuntimeConfigValue(runtime::kRuntimeSyncStreamOnDemand);
    return ret;
  }();
  if (!sync_stream_on_demand || !enable_sync_stream_on_demand) {
    device::DeviceContextKey host_key = {dst_device_tensor->GetDeviceType(), dst_device_tensor->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->SyncAllStreams();
  }

  return SyncStreamOnDemandForDeviceAddress(dst_device_tensor, src_device_tensor, stream_id);
}

/**
 * @brief Sync streams on demand, help method for copying.
 *  In case:
 *    1 dst is cpu and src is cpu, return true directly
 *    2 src is not cpu, sync src stream
 *    3 src is cpu and dst is not cpu, sync dst when stream_id is different from dst
 *  When the sync switch: sync_stream_on_demand is disabled, follow the legacy process
 *
 * @param stream_id stream id scheduled for copying
 * @return return false means sync stream failed
 */
bool SyncStreamOnDemandForDeviceAddress(const DeviceTensorPtr &dst_device_tensor,
                                        const DeviceTensorPtr &src_device_tensor, uint32_t stream_id) {
  if (src_device_tensor->GetDeviceType() == device::DeviceType::kCPU &&
      dst_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "No need sync stream since both src and dst device tensors are cpu tensor.";
    return true;
  }

  device::DeviceType device_type = src_device_tensor->GetDeviceType() != device::DeviceType::kCPU
                                     ? src_device_tensor->GetDeviceType()
                                     : dst_device_tensor->GetDeviceType();
  auto ms_context = MsContext::GetInstance();
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey host_key = {device_type, device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  auto &res_manager = host_context->device_res_manager_;
  if (src_device_tensor->GetDeviceType() != device::DeviceType::kCPU && stream_id != src_device_tensor->stream_id()) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "sync stream : " << src_device_tensor->stream_id();
    if (!res_manager->SyncStream(src_device_tensor->stream_id())) {
      return false;
    }
  }
  // protect dst device tensor
  if (dst_device_tensor->GetDeviceType() != device::DeviceType::kCPU && stream_id != dst_device_tensor->stream_id()) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "sync stream : " << dst_device_tensor->stream_id();
    if (!res_manager->SyncStream(dst_device_tensor->stream_id())) {
      return false;
    }
  }
  return true;
}

bool CopyDataForParameter(const KernelTensorPtr &dst_kernel_tensor, Tensor *const src_tensor, size_t stream_id,
                          bool *has_h2d_copy) {
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor);
  MS_EXCEPTION_IF_NULL(src_tensor);
  const auto &dst_device_tensor = dst_kernel_tensor->device_address();
  const auto &src_device_tensor = src_tensor->device_address();
  MS_EXCEPTION_IF_NULL(dst_device_tensor);
  MS_EXCEPTION_IF_NULL(src_device_tensor);
  // judge copy operation only for capture graph.
  if (has_h2d_copy != nullptr) {
    *has_h2d_copy = true;
  }
  // D2H need use sync copy, make sure that cpu ops use ptr after copy finished.
  if (dst_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    if (stream_id == SIZE_MAX && src_device_tensor->GetDeviceType() != device::DeviceType::kCPU) {
      stream_id = src_device_tensor->stream_id();
    }
    MS_LOG(DEBUG) << "Sync copy from device tensor:" << src_device_tensor << " to:" << dst_device_tensor
                  << " by stream id:" << stream_id;
    if (!SyncAllStreamForDeviceAddress(dst_device_tensor, src_device_tensor, static_cast<uint32_t>(stream_id))) {
      MS_LOG(ERROR) << "Failed to sync all stream.";
      return false;
    }
    return SyncCopy(dst_kernel_tensor.get(), src_tensor, stream_id);
  }
  // H2D use async copy.
  if (stream_id == SIZE_MAX) {
    stream_id = dst_device_tensor->stream_id();
  }
  MS_LOG(DEBUG) << "Async copy from device tensor:" << src_device_tensor << " to:" << dst_device_tensor
                << " by stream id:" << stream_id;
  auto ret = AsyncCopy(dst_kernel_tensor.get(), src_tensor, stream_id, false);
  static bool sync_copy_input = runtime::IsEnableRuntimeConfig(runtime::kRuntimeSyncCopyInput);
  if (sync_copy_input) {
    if (!SyncAllStreamForDeviceAddress(dst_device_tensor, src_device_tensor, static_cast<uint32_t>(stream_id), false)) {
      MS_LOG(ERROR) << "Failed to sync all stream.";
      return false;
    }
  }
  return ret;
}

void FreeMemoryByDeviceContext(DeviceTensor *const device_tensor, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  // The device context may be not accurate in the control flow scene, so need fetch by device name and device id.
  if ((device_context == nullptr) || (device_context->GetDeviceType() != device_tensor->GetDeviceType())) {
    const auto &new_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->GetDeviceType(), device_tensor->device_id()});
    MS_EXCEPTION_IF_NULL(new_device_context);
    new_device_context->device_res_manager_->FreeMemory(device_tensor);
  } else {
    device_context->device_res_manager_->FreeMemory(device_tensor);
  }
}

void FreeMemoryByValueNode(const std::vector<std::weak_ptr<ValueNode>> &held_by_nodes, DeviceTensor *device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->ClearHeldByNodes();

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
  if (IsSuperKernelActor(node, kernel_graph)) {
    return KernelTransformType::kSuperKernelActor;
  }

  KernelTransformType type = KernelTransformType::kUnknown;
  MS_EXCEPTION_IF_NULL(node);
  auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  MS_EXCEPTION_IF_NULL(real_node);

  if (EnableInputOptimize()) {
    if (IsGraphRootParameter(real_node, kernel_graph, host_parameters, strategy)) {
      return KernelTransformType::kGraphParameterStore;
    }
  }

  if (IsHostQueueDSActor(real_node, kernel_graph, host_parameters, strategy)) {
    type = KernelTransformType::kHostDataSourceActor;
  } else if (IsKernelActor(real_node, strategy)) {
    type = KernelTransformType::kKernelActor;
  } else if (IsInternalParameter(real_node, kernel_graph)) {
    type = KernelTransformType::kInternalParameter;
  } else if (IsPersistentDeviceTensor(real_node)) {
    type = KernelTransformType::kDeviceTensorStore;
  } else {
    // May exist the from kernel that no need link in the pynative mode.
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Invalid from kernel: " << node->DebugString();
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
    case KernelTransformType::kHostDataSourceActor:
      actor_name = actor_set_name + kHostDSActorNameSuffix;
      break;
    case KernelTransformType::kGraphParameterStore:
      actor_name = actor_set_name + kReplaceDSActorStore;
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

  if (graph_to_kernel_tensor_with_mem_blocks_.find(graph_id) == graph_to_kernel_tensor_with_mem_blocks_.end()) {
    graph_to_kernel_tensor_with_mem_blocks_.emplace(
      graph_id, std::make_shared<HashMap<kernel::KernelTensor *, KernelMemoryTraceBlockPtr>>());
  }
  kernel_tensor_to_kernel_mem_blocks_ = graph_to_kernel_tensor_with_mem_blocks_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_tensor_to_kernel_mem_blocks_);
}

void MemoryTraceManager::AddKernelMemoryTraceBlock(const KernelMemoryTraceBlockPtr &block,
                                                   const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(block->start_);
  MS_EXCEPTION_IF_NULL(block->end_);
  (*kernel_memory_trace_blocks_)[device_context].emplace_back(block);
}

const std::shared_ptr<std::map<const DeviceContext *, std::vector<MemoryTraceBlockPtr>>> &
MemoryTraceManager::GetMergeBlocks() {
  return merged_memory_trace_blocks_;
}

const std::shared_ptr<mindspore::HashMap<CNodePtr, std::vector<KernelMemoryTraceBlockPtr>>> &
MemoryTraceManager::GetAllKernelBlocksnfo() {
  return kernel_to_block_;
}

const std::shared_ptr<HashMap<kernel::KernelTensor *, KernelMemoryTraceBlockPtr>> &
MemoryTraceManager::GetKernelTensorToMemBlocksInfo() const {
  return kernel_tensor_to_kernel_mem_blocks_;
}

void MemoryTraceManager::MergeBlocks() {
  merged_memory_trace_blocks_->clear();
  for (auto &item : *kernel_memory_trace_blocks_) {
    auto &device_context = item.first;
    auto &kernel_memory_trace_blocks = item.second;
    MergeBlocksForSameDeviceContext(&kernel_memory_trace_blocks, &((*merged_memory_trace_blocks_)[device_context]));
    MS_LOG(DEBUG) << "The number of merged blocks is " << (*merged_memory_trace_blocks_)[device_context].size()
                  << ", device type: " << device_context->device_context_key().device_type_;
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
    if (EnableParallelDispatchKernel() && kernel_mem_block->mem_type_ == kOutputMem) {
      kernel_tensor_to_kernel_mem_blocks_->emplace(kernel_mem_block->kernel_tensor_, kernel_mem_block);
    }
  }
}

void MemoryTraceManager::ClearExpiredCache() {
  kernel_memory_trace_blocks_->clear();
  merged_memory_trace_blocks_->clear();
  kernel_to_block_->clear();
  if (EnableParallelDispatchKernel()) {
    kernel_tensor_to_kernel_mem_blocks_->clear();
  }
}

void MemoryTraceManager::ClearAllCache() {
  for (auto &item : graph_to_kernel_memory_trace_blocks_) {
    if (item.second) {
      item.second->clear();
    }
  }
  graph_to_kernel_memory_trace_blocks_.clear();

  for (auto &item : graph_to_merged_memory_trace_blocks_) {
    if (item.second) {
      item.second->clear();
    }
  }
  graph_to_merged_memory_trace_blocks_.clear();

  for (auto &item : graph_to_kernel_blocks_) {
    if (item.second) {
      item.second->clear();
    }
  }
  graph_to_kernel_blocks_.clear();

  for (auto &item : graph_to_kernel_tensor_with_mem_blocks_) {
    if (item.second) {
      item.second->clear();
    }
  }
  graph_to_kernel_tensor_with_mem_blocks_.clear();

  kernel_memory_trace_blocks_ = nullptr;
  merged_memory_trace_blocks_ = nullptr;
  kernel_to_block_ = nullptr;
  kernel_tensor_to_kernel_mem_blocks_ = nullptr;
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

  return flatten_tensors[input_tensor_index];
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

void UpdateDynamicShapeAndSize(tensor::Tensor *input_tensor, const KernelTensorPtr &kernel_tensor, size_t outer_index,
                               size_t inner_index) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (input_tensor == nullptr || IsEmptySequenceTensor(input_tensor)) {
    return;
  }

  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  if (!IsDynamic(kernel_tensor->GetShapeVector()) &&
      !graph_parameter_store->IsPositionDynamic(outer_index, inner_index)) {
    MS_LOG(DEBUG) << "No need to update dynamic shape and size, host shape dynamic is "
                  << IsDynamic(kernel_tensor->GetShapeVector())
                  << ", graph parameter store outer index: " << outer_index << ", inner index: " << inner_index
                  << ", dynamic is " << graph_parameter_store->IsPositionDynamic(outer_index, inner_index);
    return;
  }

  // Update shape.
  if (input_tensor->base_shape_ptr() == nullptr || (!input_tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    kernel_tensor->SetShape(input_tensor->ToAbstract()->GetShape());
    MS_LOG(DEBUG) << "Kernel tensor: " << kernel_tensor.get() << ", shape is " << kernel_tensor->GetShapeVector();
    return;
  }
  kernel_tensor->SetShape(input_tensor->base_shape_ptr());
  MS_LOG(DEBUG) << "Kernel tensor: " << kernel_tensor.get() << ", shape is " << kernel_tensor->GetShapeVector();

  // Update size.
  const auto &device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto device_format = kernel_tensor->format();
  static const std::set<Format> kNormalFormat = {
    Format::DEFAULT_FORMAT, Format::ND, Format::NCHW, Format::NHWC, Format::HWCN,
  };
  if (kNormalFormat.find(device_format) != kNormalFormat.end()) {
    auto tensor_data_size = input_tensor->DataNBytes();
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Set device address:" << device_tensor << " size from:" << device_tensor->GetSize()
      << " to:" << tensor_data_size;
    device_tensor->SetSize(tensor_data_size);
  } else {
    MS_LOG(EXCEPTION) << "Can not Update size for 5D format device address";
  }
}

void CheckValidInputForParallelDispatch(const device::DeviceAddressPtr &tensor_address, const AnfNodePtr &node) {
  // In parallel dispatch scenarios, asynchronous host-to-device copying requires the host address to be pinned memory;
  // otherwise, it can lead to deadlock issues, as aclrtMemcpyAsync includes synchronous stream behavior.
  if (ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_actor_set()) {
    MS_EXCEPTION_IF_NULL(tensor_address);
    MS_EXCEPTION_IF_NULL(node);
    bool has_pinned_allocator = (tensor_address->allocator() != nullptr) && (tensor_address->allocator()->IsPinned());
    if (tensor_address->GetDeviceType() == device::DeviceType::kCPU && !has_pinned_allocator) {
      std::string err_info =
        std::string(
          "In parallel dispatch(kernel group launch) scenarios, asynchronous host-to-device copying for network input "
          "tensor requires the host address to be pinned memory. Please change the host memory allocation method for "
          "the "
          "input tensor in the Python layer to pinned memory for input node: ") +
        node->DebugString();
      // To prevent thread exceptions from being overwritten, always prints of the first ERROR log.
      MS_LOG(ERROR) << err_info;
      MS_LOG(EXCEPTION) << err_info;
    }
  }
}

void AllocMemAndCopyForParameter(size_t outer_index, size_t inner_index, tensor::Tensor *tensor, const AID &from_aid,
                                 const AnfNodePtr &node, bool is_first_user, size_t stream_id,
                                 bool *has_h2d_copy = nullptr) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kKernelPrepareData, from_aid.Name());
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
  if (NeedRunMemTracker()) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), node->fullname_with_scope(),
                                                   from_aid.Name(), false);
  }
  // Update dynamic shape and size.
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const auto &device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (graph_parameter_store->GetUserCnt(outer_index, inner_index) == 0) {
    MS_LOG(DEBUG) << "Skip sync host to device for kernel tensor:" << kernel_tensor->ToString()
                  << " outer index:" << outer_index << " inner index:" << inner_index << " for user count:0.";
    return;
  }
  UpdateDynamicShapeAndSize(tensor, kernel_tensor, outer_index, inner_index);
  graph_parameter_store->ResetAddrRefCount(outer_index, inner_index);
  if (TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagNotUsed)) {
    kernel_tensor->IncreaseNewRefCount(from_aid.Name());
    MS_LOG(DEBUG) << from_aid.Name() << " do not use input outer index: " << outer_index
                  << ", inner index: " << inner_index << ", address: " << device_tensor
                  << " from graph parameter store.";
    return;
  }
  CheckValidInputForParallelDispatch(tensor->device_address(), node);

  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->GetDeviceType(), device_tensor->device_id()});

  if (device_tensor->GetPtr() == nullptr) {
    auto mem_type = kernel_tensor->new_ref_count() == SIZE_MAX ? memory::mem_pool::MemType::kWeight
                                                               : memory::mem_pool::MemType::kKernel;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(), mem_type, device_tensor->GetSize(),
                                                   device_tensor.get());
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
    if (!device_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed, outer index: " << outer_index << ", inner index: " << inner_index
                        << ", for kernel tensor: " << kernel_tensor->ToString();
    }
    static std::string name = "Alloc memory";
    kernel_tensor->IncreaseNewRefCount(name);
  } else {
    if (!(graph_parameter_store->GetPositionWeight(outer_index) || common::AnfAlgo::HasAbstractRef(node))) {
      MS_LOG(EXCEPTION) << "The device ptr is not nullptr, there is memory leak for outer size: " << outer_index
                        << ", inner size: " << inner_index << ", device tensor info: " << device_tensor->ToString()
                        << ", node: " << node->fullname_with_scope();
    }
  }

  static auto is_reboot_node_cb = GET_COMMON_CALLBACK(IsRebootNode, bool);
  bool skip_h2d = is_reboot_node_cb != nullptr && is_reboot_node_cb();
  if (skip_h2d && graph_parameter_store->GetPositionWeight(outer_index)) {
    return;
  }

  auto tensor_size = tensor->DataNBytes();
  if (is_first_user) {
    if (tensor_size > 0 && !CopyDataForParameter(kernel_tensor, tensor, stream_id, has_h2d_copy)) {
      MS_LOG(EXCEPTION) << "Fetch parameter async host to device failed.";
    }
  } else if (graph_parameter_store->GetAsyncMemcpyFun(outer_index, inner_index) == nullptr) {
    graph_parameter_store->SetAsyncMemcpyFun(
      outer_index, inner_index, [tensor_size, kernel_tensor, tensor, has_h2d_copy](size_t stream_id) {
        if (tensor_size > 0 && !CopyDataForParameter(kernel_tensor, tensor, stream_id, has_h2d_copy)) {
          MS_LOG(EXCEPTION) << "Fetch parameter async host to device failed.";
        }
      });
  }

  graph_parameter_store->InsertDeviceTensorIntoCallback(tensor->device_address());
}

bool IsRefOutputInTuple(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->abstract() == nullptr || !node->abstract()->isa<abstract::AbstractSequence>()) {
    return false;
  }
  const auto &seq_abs = node->abstract()->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len() || index >= seq_abs->size()) {
    return false;
  }
  const auto &sub_abs = seq_abs->elements()[index];
  MS_EXCEPTION_IF_NULL(sub_abs);
  return sub_abs->isa<abstract::AbstractRefTensor>();
}

void PrepareParameterWithCopy(const std::pair<KernelWithIndex, size_t> &parameter_index, Tensor *tensor,
                              const AID &from_aid, bool is_first_user, size_t stream_id, bool *has_h2d_copy = nullptr) {
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto outer_index = parameter_index.second;
  auto inner_index = parameter_index.first.second;
  auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  if (device_tensor == nullptr) {
    abstract::BaseShapePtr shape;
    if (tensor->base_shape_ptr() == nullptr || (!tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
      shape = tensor->ToAbstract()->GetShape();
    } else {
      shape = tensor->base_shape_ptr();
    }
    MS_EXCEPTION_IF_NULL(shape);
    auto old_addr_info_ret = graph_parameter_store->GetReleasePositionInfo({outer_index, inner_index});
    if (!old_addr_info_ret.first) {
      MS_LOG(EXCEPTION) << "Can not find info, outer index: " << outer_index << ", inner index: " << inner_index;
    }
    auto old_addr_info = old_addr_info_ret.second;
    TypePtr type = old_addr_info.first;
    MS_EXCEPTION_IF_NULL(type);
    auto device_type = graph_parameter_store->GetParameterDeviceType(outer_index, inner_index);
    auto new_device_tensor = std::make_shared<device::DeviceAddress>(device_type);
    auto new_kernel_tensor = std::make_shared<kernel::KernelTensor>(new_device_tensor, shape, type, nullptr);
    new_kernel_tensor->set_size(LongToSize(tensor->DataNBytes()));
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Refresh store device tensor, from: " << new_device_tensor.get() << ", to null,"
      << ", outer index: " << outer_index << ", inner index: " << inner_index
      << ", device type: " << device::GetDeviceNameByType(new_device_tensor->GetDeviceType());
    new_device_tensor->SetNodeIndex(old_addr_info.second.first, old_addr_info.second.second);
    new_device_tensor->set_from_persistent_mem(true);
    kernel_tensor->set_device_address(new_device_tensor);
    device_tensor = new_device_tensor;
  }

  auto front_node = parameter_index.first;
  MS_EXCEPTION_IF_NULL(front_node.first);
  AllocMemAndCopyForParameter(outer_index, inner_index, tensor, from_aid, front_node.first, is_first_user, stream_id,
                              has_h2d_copy);
  if (graph_parameter_store->GetPositionWeight(outer_index) || common::AnfAlgo::HasAbstractRef(front_node.first) ||
      IsRefOutputInTuple(front_node.first, inner_index)) {
    tensor->set_device_address(device_tensor);
    tensor->set_format(kernel_tensor->format());
    kernel_tensor->set_new_ref_count(SIZE_MAX);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Set new ref count to max for device address:" << device_tensor
                                                 << " parameter:" << front_node.first->DebugString()
                                                 << " outer index:" << outer_index << " inner index:" << inner_index;
  }
  graph_parameter_store->SetDeviceTensorPrepared(outer_index, inner_index, true);
}

void SetNodeIndexForTensorAddress(const DeviceTensorPtr &device_tensor, const DeviceTensorPtr &tensor_address,
                                  size_t outer_index, size_t inner_index) {
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  if (device_tensor != nullptr) {
    const auto &node_with_index = device_tensor->GetNodeIndex();
    tensor_address->SetNodeIndex(node_with_index.first, node_with_index.second);
  } else {
    auto old_addr_info_ret = graph_parameter_store->GetReleasePositionInfo({outer_index, inner_index});
    if (old_addr_info_ret.first) {
      auto old_addr_info = old_addr_info_ret.second;
      tensor_address->SetNodeIndex(old_addr_info.second.first, old_addr_info.second.second);
    }
  }
}

void CheckInputSize(const KernelTensorPtr &kernel_tensor, Tensor *tensor, size_t outer_index, size_t inner_index) {
  static bool sync_copy_input = runtime::IsEnableRuntimeConfig(runtime::kRuntimeSyncCopyInput) ||
                                runtime::RuntimeConf::GetInstance()->launch_blocking();
  if (sync_copy_input) {
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_EXCEPTION_IF_NULL(tensor);
    auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
    if (!graph_parameter_store->IsPositionDynamic(outer_index, inner_index)) {
      auto tmp = kernel_tensor;
      if (kernel_tensor->device_address() == nullptr) {
        tmp = graph_parameter_store->GetReleasedCheckInfo(outer_index, inner_index);
      }
      if (tmp == nullptr) {
        return;
      }
      auto device_tensor = tmp->device_address();
      // The input size for static shape and normal format remains unchanged.
      if (device_tensor && device_tensor->GetTensorStorageInfo() == nullptr &&
          (tmp->format() == DEFAULT_FORMAT || tmp->format() == ND) && device_tensor->size() != tensor->Size()) {
        MS_LOG(ERROR) << "The tensor size " << tensor->Size() << " is different from device tensor size "
                      << device_tensor->size() << ", outer index: " << outer_index << ", inner index: " << inner_index
                      << kernel_tensor->ToString();
      }
    }
  }
}

device::DeviceAddressPtr PrepareOffloadedParameter(Tensor *tensor, const KernelTensorPtr &kernel_tensor,
                                                   const device::DeviceAddressPtr &device_address) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &tensor_address = tensor->device_address();
  MS_EXCEPTION_IF_NULL(tensor_address);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(device_address);
  if (tensor->device_address()->GetDeviceType() != device::DeviceType::kCPU) {
    Pipeline::Get().WaitForward();
  }
  auto allocator = device_address->allocator();
  const auto size = device_address->GetSize();
  if (device_address->GetDeviceType() != device::DeviceType::kCPU || allocator == nullptr) {
    return tensor_address;
  }

  auto pinned_tensor_address = std::static_pointer_cast<device::DeviceAddress>(
    MakeDeviceAddress(tensor->data_type(), tensor->shape(), false, device::DeviceType::kCPU));
  pinned_tensor_address->set_allocator(allocator);
  auto pin_mem_ptr = allocator->Alloc(size, kDefaultStreamIndex);
  pinned_tensor_address->set_ptr(pin_mem_ptr);
  if (pin_mem_ptr != nullptr && tensor_address->GetPtr() != nullptr) {
    if (!UseSimulationApi()) {
      bool ret = SyncCopy(pinned_tensor_address, tensor_address, kDefaultStreamIndex);
      if (!ret) {
        MS_LOG(EXCEPTION) << "Copy from origin host ptr[" << tensor_address->GetPtr() << "] to pin memory["
                          << pin_mem_ptr << "failed, size: " << size;
      }
    }
  }
  tensor->set_device_address(pinned_tensor_address);
  kernel_tensor->set_device_address(tensor_address);
  MS_LOG(INFO) << "User pin memory for offloaded parameter.";
  return pinned_tensor_address;
}

void PrepareParameter(const std::pair<KernelWithIndex, size_t> &parameter_index, const AID &from_aid,
                      bool is_first_user, size_t stream_id, bool enable_parallel_dispatch,
                      bool *has_h2d_copy = nullptr) {
  // Check parameter prepared for concurrent
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto outer_index = parameter_index.second;
  auto inner_index = parameter_index.first.second;
  auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
  if (graph_parameter_store->GetDeviceTensorPrepared(outer_index, inner_index)) {
    if (is_first_user && enable_parallel_dispatch) {
      const auto &copy_func = graph_parameter_store->GetAsyncMemcpyFun(outer_index, inner_index);
      if (copy_func != nullptr) {
        copy_func(stream_id);
        graph_parameter_store->SetAsyncMemcpyFun(outer_index, inner_index, nullptr);
      }
    }
    return;
  }
  auto front_node = parameter_index.first;
  MS_LOG(DEBUG) << "Prepare parameter input, actor: " << from_aid.Name() << ", outer index: " << outer_index
                << ", inner index:" << inner_index << ", front node: " << front_node.first->DebugString();
  auto tensor = graph_parameter_store->FetchTensor(outer_index, front_node);
  MS_EXCEPTION_IF_NULL(tensor);
  CheckInputSize(kernel_tensor, tensor, outer_index, inner_index);
  auto tensor_address = tensor->device_address();
  if (tensor_address == nullptr) {
    // Tensor with initializer but didn't init_data yet.
    auto empty_tensor = tensor::from_spec(tensor->data_type(), tensor->shape(), device::DeviceType::kCPU);
    tensor->set_device_address(empty_tensor->device_address());
    tensor_address = empty_tensor->device_address();
  }
  auto device_tensor = kernel_tensor->device_address();

  if (graph_parameter_store->GetOffloaded(outer_index, inner_index) &&
      !graph_parameter_store->GetPinned(outer_index, inner_index)) {
    tensor_address = PrepareOffloadedParameter(tensor, kernel_tensor, device_tensor);
    graph_parameter_store->SetPinned(outer_index, inner_index, true);
    MS_LOG(DEBUG) << "Prepare offloaded parameter: " << front_node.first->fullname_with_scope();
  }
  if (tensor_address->GetDeviceType() != graph_parameter_store->GetParameterDeviceType(outer_index, inner_index)) {
    MS_LOG(DEBUG) << "tensor address:" << tensor_address->ToString() << " parameter store device type:"
                  << device::GetDeviceNameByType(
                       graph_parameter_store->GetParameterDeviceType(outer_index, inner_index))
                  << " outer index:" << outer_index << " inner index:" << inner_index;
    if (!IsContiguousStorage(tensor_address->GetTensorStorageInfo())) {
      MS_LOG(EXCEPTION) << "Not support non-contiguous heter graph input index:" << outer_index
                        << " inner index:" << inner_index << " device address:" << tensor_address->ToString()
                        << " parameter store device type:"
                        << graph_parameter_store->GetParameterDeviceType(outer_index, inner_index);
    }
    static const bool use_hyper_offload = (common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1");
    if (!(use_hyper_offload && device_tensor != nullptr && device_tensor->remote())) {
      PrepareParameterWithCopy(parameter_index, tensor, from_aid, is_first_user, stream_id, has_h2d_copy);
      return;
    }
  }
  graph_parameter_store->SetDeviceTensorPrepared(outer_index, inner_index, true);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Set new ref count to max for device address:" << tensor_address;
  kernel_tensor->set_new_ref_count(SIZE_MAX);
  if (tensor_address->GetPtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Device ptr of tensor address can not be nullptr, device type: "
                      << tensor_address->GetDeviceType() << " for parameter index:" << outer_index
                      << " inner index:" << inner_index << " device address:" << tensor_address->ToString();
  }

  if (tensor_address == device_tensor) {
    return;
  }

  // Set tensor address to kernel tensor.
  MS_LOG(DEBUG) << "Set tensor address to kernel tensor, tensor address: " << tensor_address->ToString()
                << ", old device address: " << ((device_tensor == nullptr) ? "nullptr" : device_tensor->ToString())
                << ", outer index: " << outer_index << ", inner index: " << inner_index
                << ", kernel tensor: " << kernel_tensor.get();
  SetNodeIndexForTensorAddress(device_tensor, tensor_address, outer_index, inner_index);
  if (tensor->format() != kernel_tensor->format()) {
    MS_LOG(INFO) << "Input format:" << tensor->format()
                 << " for graph parameter kernel tensor:" << kernel_tensor->ToString();
  }
  kernel_tensor->set_device_address(tensor_address);
  UpdateDynamicShapeAndSize(tensor, kernel_tensor, outer_index, inner_index);
}

KernelTensorPtr FetchParameter(const std::pair<KernelWithIndex, size_t> &parameter_index, const AID &from_aid,
                               bool is_first_user, size_t stream_id, bool enable_parallel_dispatch,
                               bool *has_h2d_copy) {
  auto front_node = parameter_index.first.first;
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto outer_index = parameter_index.second;
  auto inner_index = parameter_index.first.second;
  graph_parameter_store->CheckIndexValid(outer_index, inner_index);
  MS_LOG(DEBUG) << "Fetch parameter for actor: " << from_aid.Name() << ", front node: " << front_node->DebugString()
                << ", with index: " << parameter_index.first.second << ", addr index: " << parameter_index.second;

  // The parameter is not concurrently used, do not use lock.
  if (!graph_parameter_store->IsConcurrentlyUse(outer_index, inner_index)) {
    // Return device tensor from graph parameter store if data prepared.
    auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    if (graph_parameter_store->GetDeviceTensorPrepared(outer_index, inner_index)) {
      // parallel dispatch kernel can not support multi graph parallel execute, no need lock.
      if (is_first_user && enable_parallel_dispatch) {
        const auto &copy_func = graph_parameter_store->GetAsyncMemcpyFun(outer_index, inner_index);
        if (copy_func != nullptr) {
          copy_func(stream_id);
          graph_parameter_store->SetAsyncMemcpyFun(outer_index, inner_index, nullptr);
        }
      }
      return kernel_tensor;
    }

    PrepareParameter(parameter_index, from_aid, is_first_user, stream_id, enable_parallel_dispatch, has_h2d_copy);
    auto is_weight = graph_parameter_store->GetPositionWeight(outer_index);
    if (!is_weight && kernel_tensor->device_address() != nullptr && kernel_tensor->new_ref_count() == SIZE_MAX) {
      graph_parameter_store->InsertNonWeightRefMaxInputs(outer_index, inner_index);
    }
    return kernel_tensor;
  }

  // Return device tensor from graph parameter store if data prepared.
  static std::shared_mutex mtx;
  std::shared_lock<std::shared_mutex> read_lock(mtx);
  auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (graph_parameter_store->GetDeviceTensorPrepared(outer_index, inner_index)) {
    // parallel dispatch kernel can not support multi graph parallel execute, no need lock.
    if (is_first_user && enable_parallel_dispatch) {
      const auto &copy_func = graph_parameter_store->GetAsyncMemcpyFun(outer_index, inner_index);
      if (copy_func != nullptr) {
        copy_func(stream_id);
        graph_parameter_store->SetAsyncMemcpyFun(outer_index, inner_index, nullptr);
      }
    }
    return kernel_tensor;
  }

  read_lock.unlock();
  std::unique_lock<std::shared_mutex> write_lock(mtx);
  PrepareParameter(parameter_index, from_aid, is_first_user, stream_id, enable_parallel_dispatch, has_h2d_copy);
  auto is_weight = graph_parameter_store->GetPositionWeight(outer_index);
  if (!is_weight && kernel_tensor->device_address() != nullptr && kernel_tensor->new_ref_count() == SIZE_MAX) {
    graph_parameter_store->InsertNonWeightRefMaxInputs(outer_index, inner_index);
  }
  return kernel_tensor;
}
}  // namespace runtime
}  // namespace mindspore
