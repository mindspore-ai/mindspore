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

#include "runtime/graph_scheduler/actor/rpc/mux_recv_actor.h"

namespace mindspore {
namespace runtime {
void MuxRecvActor::SetMessageHandler() {
  MS_EXCEPTION_IF_NULL(server_);
  server_->SetMessageHandler(std::bind(&MuxRecvActor::HandleMessage, this, std::placeholders::_1));
}

MessageBase *MuxRecvActor::HandleMessage(MessageBase *const msg) {
  if (msg == nullptr) {
    return distributed::rpc::NULL_MSG;
  }

  // Save from actor url.
  from_actor_aid_ = msg->From();

  ActorDispatcher::Send(GetAID(), &MuxRecvActor::RunOpInterProcessData, msg, op_context_);
  return distributed::rpc::NULL_MSG;
}
}  // namespace runtime
}  // namespace mindspore
