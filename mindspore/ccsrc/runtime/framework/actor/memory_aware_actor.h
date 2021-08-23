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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_AWARE_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_AWARE_ACTOR_H_

#include <utility>
#include <string>
#include "runtime/framework/actor/abstract_actor.h"
#include "runtime/framework/device_tensor_store.h"

namespace mindspore {
namespace runtime {
// The actor represents a set of common memory related operations of actor.
class MemoryAwareActor : public AbstractActor {
 public:
  explicit MemoryAwareActor(const std::string &name, KernelTransformType type, const AID *recorder_aid,
                            const AID &memory_manager_aid)
      : AbstractActor(name, type, recorder_aid), memory_manager_aid_(memory_manager_aid) {}
  virtual ~MemoryAwareActor() = default;

  virtual void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {}
  virtual void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {}
  virtual void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {}

 protected:
  friend class GraphScheduler;

  // The id of memory manager actor. Send message to it for alloc and free memory.
  const AID memory_manager_aid_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_AWARE_ACTOR_H_
