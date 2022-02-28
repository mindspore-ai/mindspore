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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_SEND_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_SEND_ACTOR_H_

#include <set>
#include <vector>
#include <string>
#include <memory>
#include "runtime/graph_scheduler/actor/rpc/rpc_actor.h"

namespace mindspore {
namespace runtime {
// SendActor inherits from RpcActor and it's used to send data to other processes.
class SendActor : public RpcActor {
 public:
  explicit SendActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                     const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                     GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                     const std::set<size_t> &modifiable_ref_output_indexes)
      : RpcActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                 modifiable_ref_input_indexes, modifiable_ref_output_indexes, KernelTransformType::kSendActor) {}
  ~SendActor() override = default;

  // Set send actor's destination peer info, in another word, send actor's output.
  void SetRouteInfo(uint32_t dst_rank, const std::string &dst_role, const std::string &send_src_node_name,
                    const std::string &send_dst_node_name) override;

  // Lookup peer actors' route and create connection to them.
  bool ConnectServer();

 protected:
  // After rpc send kernel is launched, inter-process data should be sent.
  void SendOutput(OpContext<DeviceTensor> *const context) override;

 private:
  // Client only supports to send MessageBase, so build MessageBase with data and url.
  std::unique_ptr<MessageBase> BuildRpcMessage(const kernel::AddressPtr &data, const std::string &server_url);

  friend class GraphScheduler;

  // This send actor's destination peers' actor ids and route table.
  std::vector<std::string> peer_actor_ids_;
  mindspore::HashMap<std::string, std::string> peer_actor_urls_;

  std::unique_ptr<TCPClient> client_;
};

using SendActorPtr = std::shared_ptr<SendActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_SEND_ACTOR_H_
