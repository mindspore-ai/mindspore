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

#include "runtime/graph_scheduler/actor/rpc/send_actor.h"

#include <utility>

namespace mindspore {
namespace runtime {
void SendActor::SetRouteInfo(uint32_t, const std::string &, const std::string &send_src_node_name,
                             const std::string &send_dst_node_name) {
  auto peer_actor_id = send_src_node_name + kInterProcessEdgeMark + send_dst_node_name;
  peer_actor_ids_.emplace_back(peer_actor_id);
  rpc_output_node_name_.emplace_back(send_dst_node_name);
}

bool SendActor::ConnectServer() {
  client_ = std::make_unique<TCPClient>();
  MS_EXCEPTION_IF_NULL(client_);
  if (!client_->Initialize()) {
    MS_LOG(EXCEPTION) << "Failed to initialize tcp server for send actor.";
  }
  // Lookup actor addresses for each peer actor.
  for (const auto &peer_actor_id : peer_actor_ids_) {
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy_);
    auto peer_actor_address = actor_route_table_proxy_->LookupRoute(peer_actor_id);
    // If route is successfully looked up, peer_actor_address is not empty.
    std::string server_url = peer_actor_address.ip() + ":" + std::to_string(peer_actor_address.port());
    if (!client_->Connect(server_url)) {
      MS_LOG(EXCEPTION) << "Failed to connect to server of actor " << peer_actor_id << ", server_url: " << server_url;
    }
    MS_LOG(INFO) << "Successfully connect to server " << server_url;
    peer_actor_urls_[peer_actor_id] = server_url;
  }
  return true;
}

void SendActor::SendOutput(OpContext<DeviceTensor> *const context) {
  MS_ERROR_IF_NULL_WO_RET_VAL(context);
  MS_ERROR_IF_NULL_WO_RET_VAL(client_);
  // Step 1: Send data and control outputs.
  AbstractActor::SendOutput(context);

  // Step 2: Erase inter-process inputs for this sequential number.
  if (input_op_inter_process_.count(context->sequential_num_) != 0) {
    input_op_inter_process_.erase(context->sequential_num_);
  }

  // Step 3: Send input data(inter-process data is the input of the Send kernel) to peers.
  if (launch_info_.inputs_.empty()) {
    MS_LOG(ERROR) << "Send kernel has no output tensor.";
    return;
  }
  auto send_output = launch_info_.inputs_[0];
  for (const auto &peer : peer_actor_urls_) {
    std::string peer_server_url = peer.second;
    auto message = BuildRpcMessage(send_output, peer_server_url);
    MS_ERROR_IF_NULL_WO_RET_VAL(message);
    client_->SendAsync(std::move(message));
  }
}

std::unique_ptr<MessageBase> SendActor::BuildRpcMessage(const kernel::AddressPtr &data, const std::string &server_url) {
  MS_ERROR_IF_NULL_W_RET_VAL(data, nullptr);
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  MS_ERROR_IF_NULL_W_RET_VAL(message, nullptr);
  message->to = AID("", server_url);
  message->body.assign(static_cast<char *>(data->addr), data->size);
  return message;
}
}  // namespace runtime
}  // namespace mindspore
