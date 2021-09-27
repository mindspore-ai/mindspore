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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <thread>
#include <algorithm>
#include "mindrt/include/actor/op_actor.h"
#include "runtime/device/device_address.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "utils/log_adapter.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using mindspore::session::KernelWithIndex;
using tensor::TensorPtr;
using DeviceTensor = mindspore::device::DeviceAddress;

// The execution result of actor.
constexpr int kSuccess = 0;
constexpr int kFailure = 1;

enum class GraphExecutionStrategy {
  kPipeline,  // The actor running is triggered only by data.
  kStep       // The actor running need be triggered by control in addition.
};

enum class KernelTransformType {
  kUnknown,
  kDataPrepareActor,
  kDeviceDataSourceActor,
  kHostDataSourceActor,
  kKernelActor,
  kCopyActor,
  kLoopCountActor,
  kOutputActor,
  kDeviceTensorStore,
  // Internal parameter is the output of previous kernel graph which is related to the input of next kernel graph.
  kInternalParameter
};

#define SET_OPCONTEXT_FAIL_RET_WITH_ERROR(op_context, message) \
  {                                                            \
    MS_LOG(ERROR) << message;                                  \
    op_context.SetFailed(kFailure);                            \
    return;                                                    \
  }

#define SET_OPCONTEXT_SUCCESS_RET(op_context) \
  {                                           \
    op_context.SetSuccess(kSuccess);          \
    return;                                   \
  }

#define SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy, op_context, message) \
  {                                                                                  \
    if (strategy == GraphExecutionStrategy::kStep) {                                 \
      MS_LOG(EXCEPTION) << message;                                                  \
    }                                                                                \
    MS_LOG(ERROR) << message;                                                        \
    op_context.SetFailed(kFailure);                                                  \
    return;                                                                          \
  }

#define SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy, op_context, device_context, kernel_name, alloc_size) \
  {                                                                                                                \
    std::string message = "Device(id:" + std::to_string((device_context).device_context_key().device_id_) +        \
                          ") memory isn't enough and alloc failed, kernel name: " + kernel_name +                  \
                          ", alloc size: " + std::to_string(alloc_size) + "B.";                                    \
    if (strategy == GraphExecutionStrategy::kStep) {                                                               \
      MS_LOG(EXCEPTION) << message;                                                                                \
    }                                                                                                              \
    MS_LOG(ERROR) << message;                                                                                      \
    (op_context).SetFailed(kFailure);                                                                              \
    return;                                                                                                        \
  }

void ComputeThreadNums(size_t *actor_thread_num, size_t *OMP_thread_num, size_t *max_thread_num);

bool IsDeviceQueueDSActor(const AnfNodePtr &node, GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

// Host parameters are parameters of root funcgraph, in control flow, only the parameters of the root funcgraph are
// in the host data source.
bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph = nullptr,
                        const std::vector<AnfNodePtr> &host_parameters = {},
                        GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

bool IsKernelActor(const AnfNodePtr &node, GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

bool IsSwitchActor(const AnfNodePtr &node);

// The skip kernel doesn't run, it exists in the inplace optimizer.
bool IsSkippedKernelActor(const AnfNodePtr &node);

// Internal parameter is not the origin parameter of func graph, it is the output of previous kernel graph which is
// related to the input of this kernel graph.
bool IsInternalParameter(const AnfNodePtr &node, const KernelGraphPtr &graph);

// Judge whether the device tensor of the node is persistent or not.
bool IsPersistentDeviceTensor(const AnfNodePtr &node);

// Judge whether the front node is in a gather actor.
bool IsGatherActor(const AnfNodePtr &front_node,
                   const std::unordered_map<std::string, OpActor<DeviceTensor> *> &actor_name_to_actor);

// Copy data from src_device_tensor to dst_device_tensor.
bool Copy(const DeviceTensor *dst_device_tensor, const DeviceTensor *src_device_tensor);

void UpdateRefCount(DeviceTensor *const device_tensor, bool is_max_ref_count = false);
// Update the reference count of device tensor by the output index of node.
void UpdateRefCount(const AnfNodePtr &node, size_t output_idx, bool is_max_ref_count = false);

// Get front node by backend node.
AnfNodePtr FetchFrontNodeByBackendNode(const AnfNodePtr &backend_node, const KernelGraphPtr &graph);
KernelWithIndex FetchFrontNodeWithIndexByGraphOutput(const KernelWithIndex &output_with_index,
                                                     const KernelGraphPtr &graph);
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_
