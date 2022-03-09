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

#include "utils/log_adapter.h"
#include "distributed/cluster/topology/utils.h"
#include "distributed/cluster/topology/common.h"
#include "proto/topology.pb.h"
#include "distributed/cluster/topology/compute_graph_node.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
bool ComputeGraphNode::Initialize() {
  // Init the address of meta server node.
  RETURN_IF_FALSE_WITH_LOG(FillMetaServerAddress(&meta_server_addr_),
                           "Failed to init the address of meta server node.");

  // Init the TCP client.
  tcp_client_ = std::make_unique<rpc::TCPClient>();
  MS_EXCEPTION_IF_NULL(tcp_client_);
  RETURN_IF_FALSE_WITH_LOG(tcp_client_->Initialize(), "Failed to create the TCP client.");

  // Register itself to meta server node.
  RETURN_IF_FALSE_WITH_LOG(Register(), "Failed to register to the meta server node.");
  return true;
}

bool ComputeGraphNode::Finalize() {
  // Release the TCP client.
  if (tcp_client_ != nullptr) {
    const auto &server_url = meta_server_addr_.GetUrl();
    tcp_client_->Disconnect(server_url);
    tcp_client_->Finalize();
    tcp_client_.reset();
  }
  return true;
}

bool ComputeGraphNode::Register() {
  MS_EXCEPTION_IF_NULL(tcp_client_);
  const auto &server_url = meta_server_addr_.GetUrl();
  RETURN_IF_FALSE_WITH_LOG(tcp_client_->Connect(server_url),
                           "Failed to connect to the meta server node url: " << server_url);
  RegistrationMessage reg_msg;
  reg_msg.set_node_id(node_id_);

  std::string content = reg_msg.SerializeAsString();
  auto message = CreateMessage(server_url, content);
  MS_EXCEPTION_IF_NULL(message);

  tcp_client_->Send(std::move(message));
  return true;
}

bool ComputeGraphNode::Heartbeat() {
  MS_EXCEPTION_IF_NULL(tcp_client_);

  HeartbeatMessage hb_msg;
  hb_msg.set_node_id(node_id_);

  const auto &server_url = meta_server_addr_.GetUrl();
  std::string content = hb_msg.SerializeAsString();
  auto message = CreateMessage(server_url, content);
  MS_EXCEPTION_IF_NULL(message);

  tcp_client_->Send(std::move(message));
  return true;
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
