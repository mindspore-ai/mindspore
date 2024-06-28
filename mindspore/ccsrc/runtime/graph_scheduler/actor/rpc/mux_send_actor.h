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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_MUX_SEND_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_MUX_SEND_ACTOR_H_

#include <string>
#include <memory>
#include <set>

#include "runtime/graph_scheduler/actor/rpc/send_actor.h"
#include "runtime/graph_scheduler/actor/rpc/mux_recv_actor.h"

namespace mindspore {
namespace runtime {
// MuxSendActor inherits from SendActor and it's used to send data to other processes.
// MuxSendActor(Multiplexed Send Actor) can send data to different Recv Actor each time, for example, when responding to
// requests or replying to requests as a service, it only needs to reply to the caller of the service, although the
// actor may have established connections with multiple Recv Actors.
class MuxSendActor : public SendActor {
 public:
  explicit MuxSendActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                        const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                        GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                        const std::set<size_t> &modifiable_ref_output_indexes)
      : SendActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                  modifiable_ref_input_indexes, modifiable_ref_output_indexes) {}
  ~MuxSendActor() override = default;

  // Set the MuxRecvActor paired with the MuxSendActor to get the 'from url' from the MuxRecvActor.
  void set_mux_recv_actor(const MuxRecvActorPtr &mux_recv_actor) { mux_recv_actor_ = mux_recv_actor; }

 private:
  // After rpc send kernel is launched, inter-process data should be sent and can be sent to different Recv Actor each
  // time. For example, when responding to a request or replying to a request as a service, it only needs to reply to
  // the caller of the service, although the actor may have established connections with multiple Recv Actors.
  // When serving as a service, the MuxSendActor and MuxRecvActor of the server are used in pairs, and the MuxSendActor
  // needs to obtain the information(ip and port) of peer that initiates this service from the corresponding
  // MuxRecvActor to response request.
  bool LaunchKernel(OpContext<DeviceTensor> *const context, bool is_skip_launch = false) override;

  // MuxSendActor and MuxRecvActor of the server are used in pairs, and the MuxSendActor
  // needs to obtain the information(ip and port) of peer from the corresponding MuxRecvActor.
  MuxRecvActorPtr mux_recv_actor_;
};

using MuxSendActorPtr = std::shared_ptr<MuxSendActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_MUX_SEND_ACTOR_H_
