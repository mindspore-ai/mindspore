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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_ACTOR_H_

#include <vector>
#include <set>
#include <mutex>
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "runtime/hardware/device_context.h"
#ifdef ENABLE_DEBUGGER
#include "debug/data_dump/dump_utils.h"
#endif

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::kernel::KernelLaunchInfo;

// The debug actor is used to debug and dump kernel info, it gets the kernel real time execution info in the device, so
// it is synchronous and blocked.
class DebugActor : public ActorBase {
 public:
  DebugActor() : ActorBase("DebugActor") {}
  ~DebugActor() override = default;

  // The debug of each node.
  void Debug(const AnfNodePtr &node, const KernelLaunchInfo *launch_info_, const DeviceContext *device_context,
             OpContext<DeviceTensor> *const op_context, const AID *from_aid);

  // The debug of kernel graph.
  void DebugForGraph(const KernelGraphPtr &graph, const DeviceContext *device_context,
                     OpContext<DeviceTensor> *const op_context, const AID *from_aid);

  // The debug on step begin.
  void DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                        const std::vector<AnfNodePtr> &origin_parameters_order,
                        std::vector<DeviceContext *> device_contexts, OpContext<DeviceTensor> *const op_context,
                        const AID *from_aid);

  // The debug on step end.
  void DebugOnStepEnd(OpContext<DeviceTensor> *const op_context, const AID *from_aid);
  static inline uint32_t current_step{0};

 private:
  // class members
  uint32_t exec_order_ = 0;

  // Support multi-thread.
  std::mutex debug_mutex_;
  std::set<uint32_t> graph_id_sets_;
};

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_ACTOR_H_
