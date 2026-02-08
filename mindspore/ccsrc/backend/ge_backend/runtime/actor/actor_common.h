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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ACTOR_COMMON_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ACTOR_COMMON_H_

#include <string>
#include <vector>
#include <set>
#include <utility>
#include <thread>
#include <algorithm>
#include <map>
#include <memory>

#include "utils/hash_map.h"
#include "actor/op_actor.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "backend/ge_backend/utils/device_address_utils.h"
#include "include/runtime/memory/mem_pool/mem_dynamic_allocator.h"
#include "tools/profiler/profiler.h"
#include "primitive/structure_op_name.h"
#include "primitive/framework_op_name.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using mindspore::session::KernelWithIndex;
using tensor::TensorPtr;
using KernelTensor = kernel::KernelTensor;
using KernelTensorPtr = kernel::KernelTensorPtr;
using DeviceTensor = mindspore::device::DeviceAddress;
using DeviceTensorPtr = std::shared_ptr<DeviceTensor>;
using mindspore::backend::ge_backend::DeviceAddressUtils;
using mindspore::device::KernelInfo;
using AddressPtrList = std::vector<kernel::AddressPtr>;
template <typename T>
using OpContext = OpRTContext<T>;
template <typename T>
using OpActor = OpRTActor<T>;

// The execution result of actor.
constexpr int kSuccess = 0;
constexpr int kFailure = 1;

enum class GraphExecutionStrategy {
  kPipeline,                   // The actor running is triggered only by data.
  kStep,                       // The actor running need be triggered by control in addition.
  kPipelineWithExecutionOrder  // The actor running is triggered by data with the persistent execution order.
};
static const std::map<GraphExecutionStrategy, std::string> kGraphExecutionStrategyStr = {
  {GraphExecutionStrategy::kPipeline, "pipeline"},
  {GraphExecutionStrategy::kStep, "step"},
  {GraphExecutionStrategy::kPipelineWithExecutionOrder, "pipeline_with_execution_order"},
};
static const std::set<std::string> no_dyn_need_update_ops = {kDynamicGetNextV2OpName, kDynamicGetNextAscendOpName,
                                                             kGetNextOpName, kGetNextFromQueueOpName, kReceiveOpName};
const char kDataPrepareActorNameSuffix[] = "_DataPrepareActor";
const char kHostDSActorNameSuffix[] = "_HostDSActor";
const char kSuperKernelActorNameSuffix[] = "_SuperKernelActor";
const char kLoopCountActorNameSuffix[] = "_LoopCountActor";
const char kOutputActorNameSuffix[] = "_OutputActor";
const char kEntranceActorNameSuffix[] = "_EntranceActor";
const char kExitActorNameSuffix[] = "_ExitActor";
const char kStackActorNameSuffix[] = "_StackActor";
const char kCopyActorNameSignFromStore[] = "_device_tensor_store:";

enum class KernelTransformType {
  kUnknown,
  kDataPrepareActor,
  kHostDataSourceActor,
  // Super kernel actor represents the sink executing of graph which is the combination of kernels.
  kSuperKernelActor,
  kLoopCountActor,
  kOutputActor,
  kDeviceTensorStore,
  // Internal parameter is the output of previous kernel graph which is related to the input of next kernel graph.
  kInternalParameter,
  // Control flow actor type.
  kSwitchActor,
  kGatherActor,
  kEntranceActor,
  kExitActor,
  kStackActor,
};

#define SET_OPCONTEXT_FAIL_RET_WITH_ERROR(op_context, message) \
  do {                                                         \
    if ((op_context).error_info_.empty()) {                    \
      (op_context).is_error_ = true;                           \
      (op_context).error_info_ = message;                      \
    }                                                          \
    (op_context).SetFailed(kFailure);                          \
    return;                                                    \
  } while (0);

#define SET_OPCONTEXT_SUCCESS_RET(op_context) \
  do {                                        \
    (op_context).SetSuccess(kSuccess);        \
    return;                                   \
  } while (0);

#define SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy, op_context, message) \
  do {                                                                               \
    if ((strategy) == GraphExecutionStrategy::kStep) {                               \
      MS_LOG(EXCEPTION) << (message);                                                \
    }                                                                                \
    if ((op_context).error_info_.empty()) {                                          \
      (op_context).is_error_ = true;                                                 \
      (op_context).error_info_ = message;                                            \
    }                                                                                \
    (op_context).SetFailed(kFailure);                                                \
    return;                                                                          \
  } while (0);

#define SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy, op_context, kernel_name, alloc_size) \
  do {                                                                                             \
    std::string message = "#umsg#Memory not enough:#umsg#";                                        \
    auto ms_context = MsContext::GetInstance();                                                    \
    MS_EXCEPTION_IF_NULL(ms_context);                                                              \
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);                            \
    message += "Device(id:" + std::to_string(device_id) +                                          \
               ") memory isn't enough and alloc failed, kernel name: " + (kernel_name) +           \
               ", alloc size: " + std::to_string(alloc_size) + "B.";                               \
                                                                                                   \
    if ((strategy) == GraphExecutionStrategy::kStep) {                                             \
      MS_LOG(EXCEPTION) << message;                                                                \
    } else {                                                                                       \
      MS_LOG(ERROR) << message;                                                                    \
    }                                                                                              \
    if ((op_context).error_info_.empty()) {                                                        \
      (op_context).is_error_ = true;                                                               \
      (op_context).error_info_ = message;                                                          \
    }                                                                                              \
    (op_context).SetFailed(kFailure);                                                              \
    return;                                                                                        \
  } while (0);

// Encapsulate the actor APIs associated with execution.
class ActorDispatcher {
 public:
  template <typename T, typename Arg0, typename Arg1>
  static void Send(const AID &aid, void (T::*method)(Arg0), Arg1 &&arg) {
    if (is_multi_thread_execution_) {
      Async(aid, method, arg);
    } else {
      // The single thread execution doesn't need to switch threads and calls function directly.
      auto actor_manager = ActorMgr::GetActorMgrRef();
      MS_EXCEPTION_IF_NULL(actor_manager);
      auto base_actor = actor_manager->GetActor(aid);
      T *actor = static_cast<T *>(base_actor.get());
      MS_EXCEPTION_IF_NULL(actor);
      (actor->*method)(arg);
    }
  }

  template <typename T, typename... Args0, typename... Args1>
  static void Send(const AID &aid, void (T::*method)(Args0...), Args1 &&...args) {
    if (is_multi_thread_execution_) {
      auto tuple = std::make_tuple(std::forward<Args1>(args)...);
      Async(aid, method, std::move(tuple));
    } else {
      // The single thread execution doesn't need to switch threads and calls function directly.
      auto actor_manager = ActorMgr::GetActorMgrRef();
      MS_EXCEPTION_IF_NULL(actor_manager);
      auto base_actor = actor_manager->GetActor(aid);
      T *actor = static_cast<T *>(base_actor.get());
      MS_EXCEPTION_IF_NULL(actor);
      (actor->*method)(std::forward<Args1>(args)...);
    }
  }

  template <typename T, typename Arg0, typename Arg1>
  static void SendSync(const AID &aid, void (T::*method)(Arg0), Arg1 &&arg) {
    auto actor_manager = ActorMgr::GetActorMgrRef();
    MS_EXCEPTION_IF_NULL(actor_manager);
    auto base_actor = actor_manager->GetActor(aid);
    T *actor = static_cast<T *>(base_actor.get());
    MS_EXCEPTION_IF_NULL(actor);
    (actor->*method)(arg);
  }

  template <typename T, typename... Args0, typename... Args1>
  static void SendSync(const AID &aid, void (T::*method)(Args0...), Args1 &&...args) {
    auto actor_manager = ActorMgr::GetActorMgrRef();
    auto base_actor = actor_manager->GetActor(aid);
    T *actor = static_cast<T *>(base_actor.get());
    MS_EXCEPTION_IF_NULL(actor);
    (actor->*method)(std::forward<Args1>(args)...);
  }

  template <typename T, typename... Args0, typename... Args1>
  static void SendSync(OpActor<DeviceTensor> *to_actor, void (T::*method)(Args0...), Args1 &&...args) {
    T *actor = static_cast<T *>(to_actor);
    MS_EXCEPTION_IF_NULL(actor);
    (actor->*method)(std::forward<Args1>(args)...);
  }

  static void set_is_multi_thread_execution(bool is_multi_thread_execution) {
    is_multi_thread_execution_ = is_multi_thread_execution;
  }
  static bool is_multi_thread_execution() { return is_multi_thread_execution_; }

  static bool is_memory_allocation_sync() { return is_memory_allocation_sync_; }
  static void set_is_memory_allocation_sync(bool is_memory_allocation_sync) {
    is_memory_allocation_sync_ = is_memory_allocation_sync;
  }

  static bool is_memory_free_sync() { return is_memory_free_sync_; }
  static void set_is_memory_free_sync(bool is_memory_free_sync) { is_memory_free_sync_ = is_memory_free_sync; }

 private:
  ActorDispatcher() = default;
  ~ActorDispatcher() = default;
  DISABLE_COPY_AND_ASSIGN(ActorDispatcher);

  // Decide whether use the multi thread to execute actors.
  // There are scenarios with small network and data, and the performance of multi thread execution is not as good as
  // that of single thread, so single thread execution is required at this time.
  static bool is_multi_thread_execution_;

  // Decide whether alloc and free memory synchronously.
  // The memory manager actor will not send and recv message if true.
  static bool is_memory_allocation_sync_;
  static bool is_memory_free_sync_;
};

bool IsRunningFailed(const OpContext<KernelTensor> *context);

// Host parameters are parameters of root funcgraph, in control flow, only the parameters of the root funcgraph are
// in the host data source.
bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph = nullptr,
                        const std::vector<AnfNodePtr> &host_parameters = {},
                        GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

bool IsGraphRootParameter(const AnfNodePtr &node, const KernelGraphPtr &graph = nullptr,
                          const std::vector<AnfNodePtr> &host_parameters = {},
                          GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

bool IsSwitchActor(const AnfNodePtr &node);

// Judge whether skip the launch by the env MS_KERNEL_LAUNCH_SKIP.
bool IsSkippedLaunch(const CNodePtr &kernel = nullptr, const KernelGraphPtr &kernel_graph = nullptr);

// Internal parameter is not the origin parameter of func graph, it is the output of previous kernel graph which is
// related to the input of this kernel graph.
bool IsInternalParameter(const AnfNodePtr &node, const KernelGraphPtr &graph);

// Judge whether the device tensor of the node is persistent or not.
bool IsPersistentDeviceTensor(const AnfNodePtr &node);

bool IsControlFlowActor(KernelTransformType actor_type);

size_t GetDefragMemoryStepFreq();

void UpdateRefCount(const KernelTensorPtr &kernel_tensor, bool is_max_ref_count = false);
// Update the reference count of device tensor by the output index of node.
void UpdateRefCount(const AnfNodePtr &node, size_t output_idx, bool is_max_ref_count = false);

void FreeMemoryByDeviceContext(DeviceTensor *const device_tensor);
// The memory free for the pynative bprop graph which is managed by the value node.
void FreeMemoryByValueNode(const std::vector<std::weak_ptr<ValueNode>> &held_by_nodes,
                           const KernelTensorPtr &kernel_tensor);

KernelTransformType FetchKernelTransformType(const AnfNodePtr &node, const KernelGraphPtr &graph,
                                             const std::vector<AnfNodePtr> &host_parameters = {},
                                             GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);
std::string FetchActorName(KernelTransformType kernel_type, const std::string &actor_set_name,
                           const AnfNodePtr &node = nullptr, const KernelGraphPtr &graph = nullptr);

// Fetch the input indexes which may be modified that exist in the input ref parameter.
std::set<size_t> FetchModifiableRefInputIndex(const CNodePtr &node);
// Fetch the output indexes which may be modified that exist in the ref node.
std::set<size_t> FetchModifiableRefOutputIndex(const CNodePtr &node, const KernelGraphPtr &graph);

std::string GetActorIdByKernel(const AnfNodePtr &node);
std::string GenerateActorIdByKernel(const AnfNodePtr &node);

// GetThe repeat device tensor index.
mindspore::HashMap<size_t, size_t> GetRepeatDeviceAddressIndexPair(const std::vector<KernelTensorPtr> &kernel_tensors);

// Check a graph is from inference phase.
bool IsInferPhase(const std::string &phase);
TensorPtr FetchInputTensorByArg(const VectorRef &args, size_t arg_index, const KernelWithIndex &front_node);
bool IsEmptySequenceTensor(tensor::Tensor *tensor);
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ACTOR_COMMON_H_
