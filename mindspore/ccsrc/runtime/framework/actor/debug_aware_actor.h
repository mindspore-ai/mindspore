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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_AWARE_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_AWARE_ACTOR_H_

#include <string>
#include "runtime/framework/actor/memory_aware_actor.h"

namespace mindspore {
namespace runtime {
// The actor represents a set of common debug related operations of actor.
class DebugAwareActor : public MemoryAwareActor {
 public:
  explicit DebugAwareActor(const std::string &name) : MemoryAwareActor(name) {}
  virtual ~DebugAwareActor() = default;
  virtual void SendDebugReq(OpContext<DeviceTensor> *const context) {}
  virtual void OnDebugFinish(OpContext<DeviceTensor> *const context) {}
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_AWARE_ACTOR_H_
