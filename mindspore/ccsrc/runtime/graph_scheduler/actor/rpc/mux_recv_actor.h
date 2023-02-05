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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_MUX_RECV_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_MUX_RECV_ACTOR_H_

#include <string>
#include <memory>
#include <set>

#include "runtime/graph_scheduler/actor/rpc/recv_actor.h"

namespace mindspore {
namespace runtime {
// MuxRecvActor inherits from RecvActor and it's used to receive data from other processes.
// MuxRecvActor(Multiplexed Recv Actor) can receive data from different processes, for example, when responding to
// requests as a service, it could receive requests from different processes and process them serially.
class MuxRecvActor : public RecvActor {
 public:
  explicit MuxRecvActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                        const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                        GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                        const std::set<size_t> &modifiable_ref_output_indexes)
      : RecvActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                  modifiable_ref_input_indexes, modifiable_ref_output_indexes) {}
  ~MuxRecvActor() override = default;

  // Get the from actor aid of received message.
  const AID &from_actor_aid() const { return from_actor_aid_; }

  // Clear resource of mux recv actor.
  void Clear() override;

  // Stop mux recv actor gracefully.
  void Finalize() override;

  // Stop rpc communication to avoid dead lock after exception is thrown.
  void StopRpcAtException() override;

 private:
  // Set the message handler of the server.
  void SetMessageHandler() override;

  // The message callback of the tcp server.
  MessageBase *HandleMessage(MessageBase *const msg);

  // Parse finalize command message from received message.
  void ParseFinalizeReqData(size_t data_len, const MessageBase *const msg, bool *need_finalize) override;

  // Record the from actor aid when receive a message;
  AID from_actor_aid_;

  // Whether the actor is finalized_
  std::atomic_bool finalized_{false};

  uint32_t recv_finalize_msg_cnt_{0};
};

using MuxRecvActorPtr = std::shared_ptr<MuxRecvActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_MUX_RECV_ACTOR_H_
