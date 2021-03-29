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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_INTERFACE_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_INTERFACE_ACTOR_H_

#include <utility>
#include <string>
#include "mindrt/include/actor/op_actor.h"
#include "runtime/framework/device_tensor_store.h"

namespace mindspore {
namespace runtime {
// The actor represents a set of common memory related operations of actor.
class MemoryInterfaceActor : public OpActor<DeviceTensor> {
 public:
  explicit MemoryInterfaceActor(std::string name) : OpActor(name) {}
  virtual ~MemoryInterfaceActor() = default;
  virtual void AllocateMemory(OpContext<DeviceTensor> *context) = 0;
  virtual void FreeMemory(OpContext<DeviceTensor> *context) = 0;
  virtual void OnMemoryAllocFinish(OpContext<DeviceTensor> *context) = 0;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_INTERFACE_ACTOR_H_
