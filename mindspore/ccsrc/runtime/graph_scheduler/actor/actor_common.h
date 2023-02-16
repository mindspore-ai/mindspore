/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_

#include <string>
#include <vector>
#include <set>
#include <utility>
#include <thread>
#include <algorithm>
#include <map>
#include <memory>
#include "utils/hash_map.h"
#include "mindrt/include/actor/op_actor.h"
#include "runtime/device/device_address.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_graph.h"
#include "utils/log_adapter.h"
#include "ir/tensor.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/hardware/device_context_manager.h"
#include "common/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace runtime {
using mindspore::session::KernelWithIndex;
using tensor::TensorPtr;
using DeviceTensor = mindspore::device::DeviceAddress;
using mindspore::device::DeviceContext;
using mindspore::device::KernelInfo;

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

const char kDataPrepareActorNameSuffix[] = "_DataPrepareActor";
const char kHostDSActorNameSuffix[] = "_HostDSActor";
const char kDeviceDSActorNameSuffix[] = "_DeviceDSActor";
const char kSuperKernelActorNameSuffix[] = "_SuperKernelActor";
const char kLoopCountActorNameSuffix[] = "_LoopCountActor";
const char kOutputActorNameSuffix[] = "_OutputActor";
const char kEntranceActorNameSuffix[] = "_EntranceActor";
const char kExitActorNameSuffix[] = "_ExitActor";
const char kStackActorNameSuffix[] = "_StackActor";
const char kFusionActorNameSuffix[] = "_FusionActor";
const char kMemoryAllocActorNameSuffix[] = "_MemoryAllocActor";
const char kMemoryFreeActorNameSuffix[] = "_MemoryFreeActor";
const char kCopyActorNameSignFromStore[] = "_device_tensor_store:";
const char kMemSwapInActorNameSuffix[] = "_MemorySwapInActor";
const char kMemSwapOutActorNameSuffix[] = "_MemorySwapOutActor";
const char kMemSwapActorNamePrefix[] = "MemorySwapActor_";

enum class KernelTransformType {
  kUnknown,
  kDataPrepareActor,
  kDeviceDataSourceActor,
  kHostDataSourceActor,
  kKernelActor,
  kCustomActor,
  // Super kernel actor represents the sink executing of graph which is the combination of kernels.
  kSuperKernelActor,
  kCopyActor,
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
  // RPC actor type.
  kSendActor,
  kRecvActor,
  // Fusion actor type.
  kFusionActor,
  // Memory actor type.
  kMemoryAllocActor,
  kMemoryFreeActor,
  kMemorySwapActor
};

#define SET_OPCONTEXT_FAIL_RET_WITH_ERROR(op_context, message) \
  do {                                                         \
    (op_context).error_info_ = message;                        \
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
    (op_context).error_info_ = message;                                              \
    (op_context).SetFailed(kFailure);                                                \
    return;                                                                          \
  } while (0);

#define SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy, op_context, device_context, kernel_name, alloc_size) \
  do {                                                                                                             \
    std::string message = "";                                                                                      \
    if ((device_context).device_context_key().device_name_ == "CPU") {                                             \
      message = "Memory isn't enough and alloc failed, kernel name: " + (kernel_name) +                            \
                ", alloc size: " + std::to_string(alloc_size) + "B.";                                              \
    } else {                                                                                                       \
      message = "Device(id:" + std::to_string((device_context).device_context_key().device_id_) +                  \
                ") memory isn't enough and alloc failed, kernel name: " + (kernel_name) +                          \
                ", alloc size: " + std::to_string(alloc_size) + "B.";                                              \
    }                                                                                                              \
    if ((strategy) == GraphExecutionStrategy::kStep) {                                                             \
      MS_LOG(EXCEPTION) << message;                                                                                \
    }                                                                                                              \
    (op_context).error_info_ = message;                                                                            \
    (op_context).SetFailed(kFailure);                                                                              \
    return;                                                                                                        \
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
  static void Send(const AID &aid, void (T::*method)(Args0...), Args1 &&... args) {
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
  static void SendSync(const AID &aid, void (T::*method)(Args0...), Args1 &&... args) {
    auto actor_manager = ActorMgr::GetActorMgrRef();
    auto base_actor = actor_manager->GetActor(aid);
    T *actor = static_cast<T *>(base_actor.get());
    MS_EXCEPTION_IF_NULL(actor);
    (actor->*method)(std::forward<Args1>(args)...);
  }

  template <typename T, typename... Args0, typename... Args1>
  static void SendSync(OpActor<DeviceTensor> *to_actor, void (T::*method)(Args0...), Args1 &&... args) {
    T *actor = static_cast<T *>(to_actor);
    MS_EXCEPTION_IF_NULL(actor);
    (actor->*method)(std::forward<Args1>(args)...);
  }

  static void set_is_multi_thread_execution(bool is_multi_thread_execution) {
    is_multi_thread_execution_ = is_multi_thread_execution;
  }

  static bool is_memory_allocation_sync() { return is_memory_allocation_sync_; }
  static void set_is_memory_allocation_sync(bool is_memory_allocation_sync) {
    is_memory_allocation_sync_ = is_memory_allocation_sync;
  }

  static bool is_memory_free_sync() { return is_memory_free_sync_; }
  static void set_is_memory_free_sync(bool is_memory_free_sync) { is_memory_free_sync_ = is_memory_free_sync; }

  // The first five executions are for warm-up, the next five executions are statistics of multi thread execution time,
  // and the next next five executions are statistics of single thread execution time. The first 30 step which do search
  // if there are cpu kernels.
  static constexpr size_t kMultiThreadExecutionCountBegin{31};
  static constexpr size_t kMultiThreadExecutionCountEnd{40};
  static constexpr size_t kSingleThreadExecutionCountBegin{41};
  static constexpr size_t kSingleThreadExecutionCountEnd{50};
  // The single thread execution constraint.
  static constexpr size_t kSingleThreadExecutionActorMaxNum{100};

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

bool IsRunningFailed(const OpContext<DeviceTensor> *context);

void ComputeThreadNums(size_t *actor_thread_num, size_t *actor_and_kernel_thread_num);

bool IsDeviceQueueDSActor(const AnfNodePtr &node, GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

// Host parameters are parameters of root funcgraph, in control flow, only the parameters of the root funcgraph are
// in the host data source.
bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph = nullptr,
                        const std::vector<AnfNodePtr> &host_parameters = {},
                        GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

bool IsCustomActor(const AnfNodePtr &node);

bool IsKernelActor(const AnfNodePtr &node, GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

bool IsSwitchActor(const AnfNodePtr &node);

// The skip kernel doesn't run, it exists in the inplace optimizer.
bool IsSkippedKernelActor(const AnfNodePtr &node);

bool IsRpcActor(const AnfNodePtr &node);

// Internal parameter is not the origin parameter of func graph, it is the output of previous kernel graph which is
// related to the input of this kernel graph.
bool IsInternalParameter(const AnfNodePtr &node, const KernelGraphPtr &graph);

// Judge whether the device tensor of the node is persistent or not.
bool IsPersistentDeviceTensor(const AnfNodePtr &node);

bool IsControlFlowActor(KernelTransformType actor_type);

bool IsMemoryActor(KernelTransformType actor_type);

// Judge whether skip the launch by the env MS_KERNEL_LAUNCH_SKIP.
bool IsSkippedLaunch(const CNodePtr &kernel, const KernelGraphPtr &kernel_graph);

// Copy data from src_device_tensor to dst_device_tensor.
bool Copy(const DeviceTensor *dst_device_tensor, const DeviceTensor *src_device_tensor);

void UpdateRefCount(DeviceTensor *const device_tensor, bool is_max_ref_count = false);
// Update the reference count of device tensor by the output index of node.
void UpdateRefCount(const AnfNodePtr &node, size_t output_idx, bool is_max_ref_count = false);

void FreeMemoryByDeviceContext(DeviceTensor *const device_tensor, const DeviceContext *device_context);
// The memory free for the pynative bprop graph which is managed by the value node.
void FreeMemoryByValueNode(const std::vector<std::weak_ptr<ValueNode>> &held_by_nodes, DeviceTensor *device_tensor);

KernelTransformType FetchKernelTransformType(const AnfNodePtr &node, const KernelGraphPtr &graph,
                                             const std::vector<AnfNodePtr> &host_parameters = {},
                                             GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);
std::string FetchActorName(KernelTransformType kernel_type, const std::string &actor_set_name,
                           const AnfNodePtr &node = nullptr, const KernelGraphPtr &graph = nullptr);

// Fetch the input indexes which may be modified that exist in the input ref parameter.
std::set<size_t> FetchModifiableRefInputIndex(const CNodePtr &node);
// Fetch the output indexes which may be modified that exist in the ref node.
std::set<size_t> FetchModifiableRefOutputIndex(const CNodePtr &node, const KernelGraphPtr &graph);

// Check whether this process is parameter server and enable embedding cache.
bool is_embedding_cache_server();
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_
