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

#include "runtime/graph_scheduler/actor/memory/memory_free_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"

namespace mindspore {
namespace runtime {
void MemoryFreeActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(somas_info_);
  MS_EXCEPTION_IF_CHECK_FAIL((!device_contexts_.empty()), "The device context doesn't exist.");
  MS_LOG(DEBUG) << GetAID().Name() << " free memory: " << somas_info_->base_address_;

  if (ActorDispatcher::is_memory_free_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeSomasMemory, somas_info_,
                              device_contexts_[0], context, GetAID());
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeSomasMemory, somas_info_, device_contexts_[0],
                          context, GetAID());
  }
}
}  // namespace runtime
}  // namespace mindspore
