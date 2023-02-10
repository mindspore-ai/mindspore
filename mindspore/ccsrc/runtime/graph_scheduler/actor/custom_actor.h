/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CUSTOM_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CUSTOM_ACTOR_H_

#include <string>
#include <memory>
#include <vector>
#include "runtime/graph_scheduler/actor/memory_aware_actor.h"
#include "runtime/hardware/device_context.h"
#include "ir/anf.h"

namespace mindspore {
namespace runtime {
class CustomActor : public MemoryAwareActor {
 public:
  CustomActor(const std::string &name, const AnfNodePtr &kernel, const device::DeviceContext *device_context,
              const AID &memory_manager_aid)
      : MemoryAwareActor(name, KernelTransformType::kCustomActor, nullptr, memory_manager_aid), kernel_(kernel) {
    device_contexts_.push_back(device_context);
  }
  ~CustomActor() override = default;

  // The memory related operation interface.
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;

  const AnfNodeWeakPtr &kernel() const { return kernel_; }

 protected:
  void Run(OpContext<DeviceTensor> *const context) override;
  void Init() override;

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;

  // The info of kernel.
  AnfNodeWeakPtr kernel_;
  AnfUtils::CustomActorCallback custom_func_ = {};
  GraphExecutionStrategy strategy_{GraphExecutionStrategy::kPipeline};
  // The device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
  // The device tensors for memory free.
  std::vector<DeviceTensor *> memory_free_list_;
};

using CustomActorPtr = std::shared_ptr<CustomActor>;
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CUSTOM_ACTOR_H_
