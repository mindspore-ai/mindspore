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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_FREE_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_FREE_ACTOR_H_

#include <string>
#include <memory>
#include "runtime/graph_scheduler/actor/memory_aware_actor.h"

namespace mindspore {
namespace runtime {
using mindspore::session::SomasInfo;

// The memory free actor is used to free memory of the whole graph at the end of graph running.
class MemoryFreeActor : public MemoryAwareActor {
 public:
  MemoryFreeActor(const std::string &name, const AID &memory_manager_aid, SomasInfo *somas_info,
                  const DeviceContext *device_context)
      : MemoryAwareActor(name, KernelTransformType::kMemoryFreeActor, nullptr, memory_manager_aid),
        somas_info_(somas_info) {
    (void)device_contexts_.emplace_back(device_context);
  }
  ~MemoryFreeActor() override = default;

  // The memory related operation interface.
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;

  // Get the member.
  SomasInfo *somas_info() const { return somas_info_; }

 protected:
  void Run(OpContext<DeviceTensor> *const context) override { PostRun(context); }

 private:
  friend class SchedulerHelper;

  SomasInfo *somas_info_;
};

using MemoryFreeActorPtr = std::shared_ptr<MemoryFreeActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_FREE_ACTOR_H_
