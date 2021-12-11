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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SUPER_KERNEL_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SUPER_KERNEL_ACTOR_H_

#include <string>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include "runtime/framework/actor/debug_aware_actor.h"
#include "runtime/framework/actor/actor_common.h"
#include "runtime/hardware/device_context.h"
#include "ir/anf.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceAddress;
using mindspore::device::DeviceContext;

// The Super kernel actor is used to represent the sink executing of graph which is the combination of kernels.
class SuperKernelActor : public DebugAwareActor {
 public:
  SuperKernelActor(const std::string &name, const KernelGraphPtr &graph, const DeviceContext *device_context,
                   const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid)
      : DebugAwareActor(name, KernelTransformType::kSuperKernelActor, recorder_aid, memory_manager_aid, debug_aid),
        graph_(graph) {
    (void)device_contexts_.emplace_back(device_context);
  }
  ~SuperKernelActor() override = default;

  void Init() override;

  size_t FetchInputNodePosition(const AnfNodePtr &intput_node);

  // The debug related operation interface.
  void SendDebugReq(OpContext<DeviceTensor> *const context) override;

  const KernelGraphPtr &graph() const { return graph_; }

 protected:
  void Run(OpContext<DeviceTensor> *const context) override;
  // The input may come from the control actor, so need free the input memory by the dynamic ref count.
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  bool CopyInputData(const OpContext<DeviceTensor> *context);

  KernelGraphPtr graph_;

  std::map<AnfNodePtr, DeviceAddress *> ref_node_addr_map_;

  // The lists of device tensors which need free by dynamic ref count, will be cleared at the end of step.
  std::vector<std::vector<DeviceTensor *>> memory_free_lists_;
};

using SuperKernelActorPtr = std::shared_ptr<SuperKernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
