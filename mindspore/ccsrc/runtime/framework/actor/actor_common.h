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

#include <utility>
#include "mindrt/include/actor/op_actor.h"
#include "backend/session/kernel_graph.h"
#include "utils/log_adapter.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using tensor::TensorPtr;

// The execution result of actor.
constexpr int kSuccess = 0;
constexpr int kFailure = 1;

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

// Get the max available thread number of system.
int64_t GetMaxThreadNum();

bool IsDeviceQueueDSActor(const AnfNodePtr &node);
bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph = nullptr,
                        const TensorPtr &tensor = nullptr);
bool IsKernelActor(const AnfNodePtr &node);

// Internal parameter is not the origin parameter of func graph, it is the output of previous kernel graph which is
// related to the input of this kernel graph.
bool IsInternalParameter(const AnfNodePtr &node, const KernelGraphPtr &graph);

// Judge whether the device tensor of the node is persistent or not.
bool IsPersistentDeviceTensor(const AnfNodePtr &node);
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_
