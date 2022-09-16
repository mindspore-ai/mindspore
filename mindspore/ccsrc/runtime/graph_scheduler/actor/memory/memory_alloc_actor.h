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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_ALLOC_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_ALLOC_ACTOR_H_

#include <string>
#include <memory>
#include <vector>
#include "runtime/graph_scheduler/actor/memory_aware_actor.h"

namespace mindspore {
namespace runtime {
using mindspore::session::SomasInfo;

// The memory alloc actor is used to alloc memory of the whole graph at the begin of graph running.
class MemoryAllocActor : public MemoryAwareActor {
 public:
  MemoryAllocActor(const std::string &name, const AID &memory_manager_aid, SomasInfo *somas_info,
                   const DeviceContext *device_context)
      : MemoryAwareActor(name, KernelTransformType::kMemoryAllocActor, nullptr, memory_manager_aid),
        somas_info_(somas_info),
        created_device_tensor_(nullptr) {
    (void)device_contexts_.emplace_back(device_context);
  }
  // MemoryFreeActor will free the ptr, so need set nullptr in the destructor.
  ~MemoryAllocActor() override { created_device_tensor_->set_ptr(nullptr); }

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  // The processing after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  // Get the member.
  SomasInfo *somas_info() const { return somas_info_; }

 protected:
  void Init() override;
  void Run(OpContext<DeviceTensor> *const context) override { SendMemoryAllocReq(context); }

 private:
  friend class SchedulerHelper;

  SomasInfo *somas_info_;

  DeviceTensorPtr created_device_tensor_;
  std::vector<DeviceTensor *> memory_alloc_list_;
};

using MemoryAllocActorPtr = std::shared_ptr<MemoryAllocActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_ALLOC_ACTOR_H_
